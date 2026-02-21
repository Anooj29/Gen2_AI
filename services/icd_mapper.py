import pandas as pd
import difflib
import os
from pathlib import Path
from typing import List, Dict, Optional
from services.llm import call_llama

# Global variable to store loaded ICD data
icd_data = None
icd_data_loaded = False
original_desc_col = None  # Store the description column name

def _load_icd_data():
    """Load ICD-10 dataset. Handles path resolution and errors gracefully."""
    global icd_data, icd_data_loaded, original_desc_col
    
    if icd_data_loaded:
        return icd_data
    
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent / "data" / "icd10.csv",  # services/data/icd10.csv
        Path("data/icd10.csv"),
        Path("services/data/icd10.csv"),
    ]
    
    csv_path = None
    for path in possible_paths:
        if path.exists():
            csv_path = path
            break
    
    if not csv_path:
        print(f"Warning: ICD-10 CSV file not found. Tried paths: {[str(p) for p in possible_paths]}")
        icd_data_loaded = True  # Mark as loaded to avoid repeated attempts
        return None
    
    try:
        icd_data = pd.read_csv(csv_path)
        
        # Try to find columns - handle different possible column names
        # Expected: CODE, SHORT DESCRIPTION, LONG DESCRIPTION
        code_col = None
        short_desc_col = None
        long_desc_col = None
        
        # Look for CODE column
        for col in icd_data.columns:
            col_lower = col.lower()
            if col_lower == "code" or "code" in col_lower:
                code_col = col
                break
        
        # Look for description columns
        for col in icd_data.columns:
            col_lower = col.lower()
            if "short" in col_lower and ("description" in col_lower or "desc" in col_lower):
                short_desc_col = col
            elif "long" in col_lower and ("description" in col_lower or "desc" in col_lower):
                long_desc_col = col
            elif not short_desc_col and ("description" in col_lower or "desc" in col_lower):
                # Use as both if only one description column exists
                short_desc_col = col
                long_desc_col = col
        
        if code_col and (short_desc_col or long_desc_col):
            # Create normalized columns for matching
            desc_col = long_desc_col if long_desc_col else short_desc_col
            
            # Create normalized lowercase columns for matching
            if short_desc_col:
                icd_data["short_desc_lower"] = icd_data[short_desc_col].astype(str).str.lower()
            else:
                icd_data["short_desc_lower"] = icd_data[desc_col].astype(str).str.lower()
            
            icd_data["long_desc_lower"] = icd_data[desc_col].astype(str).str.lower()
            icd_data["code"] = icd_data[code_col].astype(str)
            original_desc_col = desc_col  # Store globally for later use
            
            icd_data_loaded = True
            return icd_data
        else:
            print(f"Warning: Could not find description/code columns in ICD CSV. Found columns: {list(icd_data.columns)}")
            icd_data_loaded = True
            return None
            
    except Exception as e:
        print(f"Error loading ICD-10 CSV: {str(e)}")
        icd_data_loaded = True
        return None


def similarity_score(a: str, b: str) -> float:
    """Calculate similarity score between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def _enhance_diagnosis_with_llm(diagnosis: str) -> Optional[str]:
    """Use LLM to normalize/expand diagnosis for better ICD matching."""
    prompt = f"""
You are a medical coding assistant. Convert this diagnosis into a standardized medical term that would match ICD-10 codes.

Input diagnosis: {diagnosis}

IMPORTANT RULES:
- Output ONLY the standardized medical term (no explanations, no JSON, no extra text).
- Use standard medical terminology.
- Keep it concise (2-5 words typically).
- Examples:
  - "cold" -> "Acute upper respiratory infection"
  - "fever and cough" -> "Acute respiratory infection"
  - "chest pain" -> "Chest pain"

Standardized term:
"""
    
    try:
        result = call_llama(prompt, expect_json=False)
        if result and isinstance(result, str) and len(result.strip()) > 0:
            return result.strip()
    except Exception:
        pass
    
    return None


def map_icd10(diagnosis_list: List[str], top_k: int = 3, use_llm_enhancement: bool = True) -> List[Dict]:
    """
    Map possible diagnoses to top ICD-10 codes.
    Returns ranked results with confidence score.
    
    Args:
        diagnosis_list: List of diagnosis strings
        top_k: Number of top matches to return per diagnosis
        use_llm_enhancement: Whether to use LLM to normalize diagnoses before matching
    """
    if not diagnosis_list:
        return []
    
    # Load ICD data
    icd_data = _load_icd_data()
    if icd_data is None:
        # If CSV not available, try LLM-based mapping
        return _map_icd10_with_llm(diagnosis_list, top_k)
    
    results = []
    
    for diagnosis in diagnosis_list:
        if not diagnosis or not diagnosis.strip():
            continue
            
        diagnosis_original = diagnosis.strip()
        diagnosis_lower = diagnosis_original.lower()
        
        # Try LLM enhancement for better matching
        enhanced_diagnosis = None
        if use_llm_enhancement:
            enhanced_diagnosis = _enhance_diagnosis_with_llm(diagnosis_original)
            if enhanced_diagnosis:
                diagnosis_lower = enhanced_diagnosis.lower()
        
        # Normalize diagnosis: remove common words that don't help matching
        stop_words = {"acute", "chronic", "unspecified", "other", "and", "the", "a", "an"}
        diagnosis_words = [w for w in diagnosis_lower.split() if w not in stop_words and len(w) > 2]
        diagnosis_words_set = set(diagnosis_words)
        diagnosis_key_terms = " ".join(diagnosis_words)
        
        # Key medical terms for fast pre-filtering
        key_medical_terms = {"infection", "fever", "pain", "cough", "injury", "fracture", 
                            "disease", "syndrome", "disorder", "inflammation"}
        diagnosis_has_medical_term = any(term in diagnosis_lower for term in key_medical_terms)
        
        # FAST PRE-FILTER: Use vectorized string operations to filter candidates
        # Only check rows that have at least one matching word (much faster)
        if len(diagnosis_words) > 0:
            # Create a mask for rows that might match
            word_mask = icd_data["long_desc_lower"].str.contains("|".join(diagnosis_words[:3]), case=False, na=False, regex=True)
            # Also check short desc
            word_mask = word_mask | icd_data["short_desc_lower"].str.contains("|".join(diagnosis_words[:3]), case=False, na=False, regex=True)
            # If medical term present, also check for that
            if diagnosis_has_medical_term:
                medical_mask = icd_data["long_desc_lower"].str.contains("|".join(key_medical_terms), case=False, na=False, regex=True)
                word_mask = word_mask | medical_mask
            
            # Filter to only potential matches (reduces search space dramatically)
            candidate_rows = icd_data[word_mask].copy()
        else:
            # If no meaningful words, check a limited sample
            candidate_rows = icd_data.head(500).copy()
        
        # Limit candidates to reasonable number for performance (prioritize quality)
        if len(candidate_rows) > 500:
            # Take first 500 candidates (they're already filtered by relevance)
            candidate_rows = candidate_rows.head(500)
        
        scored_matches = []
        
        # Now only iterate through pre-filtered candidates
        for idx, row in candidate_rows.iterrows():
            short_desc = str(row["short_desc_lower"]).lower()
            long_desc = str(row["long_desc_lower"]).lower()
            code = str(row["code"])
            
            if not code or code == "nan":
                continue
            
            # Fast word overlap check first (cheapest)
            desc_words = set((short_desc + " " + long_desc).split())
            desc_words_clean = {w for w in desc_words if w not in stop_words and len(w) > 2}
            
            if len(diagnosis_words_set) > 0:
                word_overlap = len(diagnosis_words_set & desc_words_clean) / len(diagnosis_words_set)
            else:
                word_overlap = 0
            
            # Skip if no word overlap at all (fast exit)
            if word_overlap < 0.1:
                continue
            
            # Only calculate expensive similarity scores if word overlap is promising
            score_short = similarity_score(diagnosis_lower, short_desc)
            score_long = similarity_score(diagnosis_lower, long_desc)
            
            # Quick key terms check
            short_desc_clean = " ".join([w for w in short_desc.split() if w not in stop_words and len(w) > 2])
            long_desc_clean = " ".join([w for w in long_desc.split() if w not in stop_words and len(w) > 2])
            score_key_terms = max(
                similarity_score(diagnosis_key_terms, short_desc_clean),
                similarity_score(diagnosis_key_terms, long_desc_clean)
            )
            
            # Check if any key medical terms match (boost score)
            medical_term_match = diagnosis_has_medical_term and any(term in (short_desc + " " + long_desc) for term in key_medical_terms)
            
            # Combined score with multiple factors
            base_score = max(score_short, score_long, score_key_terms)
            combined_score = base_score * 0.5 + word_overlap * 0.4 + score_key_terms * 0.1
            
            # Boost if medical terms match
            if medical_term_match:
                combined_score = min(1.0, combined_score + 0.1)
            
            # Lower threshold and require at least some word overlap for better recall
            if combined_score > 0.25 or (word_overlap > 0.3 and combined_score > 0.15):
                # Get description from stored column
                global original_desc_col
                desc_col = original_desc_col
                if desc_col is None:
                    desc_cols = [col for col in icd_data.columns if "description" in col.lower() or "desc" in col.lower()]
                    desc_col = desc_cols[0] if desc_cols else None
                
                if desc_col and desc_col in row:
                    description = str(row[desc_col])
                else:
                    description = str(long_desc) if long_desc else ""
                
                scored_matches.append({
                    "diagnosis": diagnosis_original,
                    "enhanced_diagnosis": enhanced_diagnosis if enhanced_diagnosis else None,
                    "icd_code": code,
                    "description": description,
                    "confidence_score": round(combined_score, 3)
                })
                
                # Early exit if we have enough high-confidence matches
                if len(scored_matches) >= top_k * 2 and combined_score > 0.7:
                    break
        
        # Sort best matches
        scored_matches = sorted(scored_matches, key=lambda x: x["confidence_score"], reverse=True)
        
        # Deduplicate by ICD code (keep highest confidence)
        seen_codes = {}
        for match in scored_matches:
            code = match["icd_code"]
            if code and code != "nan":
                if code not in seen_codes or match["confidence_score"] > seen_codes[code]["confidence_score"]:
                    seen_codes[code] = match
        
        # Return top matches
        top_matches = list(seen_codes.values())[:top_k]
        results.extend(top_matches)
    
    return results


def _map_icd10_with_llm(diagnosis_list: List[str], top_k: int = 3) -> List[Dict]:
    """Fallback: Use LLM to suggest ICD-10 codes when CSV is not available."""
    prompt = f"""
You are a medical coding assistant. For each diagnosis, provide the most likely ICD-10 code.

IMPORTANT RULES:
- Output MUST be valid JSON only (no extra text, no markdown).
- Do NOT add explanations, comments, or notes.
- Start your response with {{ and end with }}.

JSON FORMAT:

{{
  "icd_mappings": [
    {{
      "diagnosis": "Acute upper respiratory infection",
      "icd_code": "J06.9",
      "description": "Acute upper respiratory infection, unspecified",
      "confidence_score": 0.85
    }}
  ]
}}

Diagnoses to map:
{chr(10).join(f"- {d}" for d in diagnosis_list)}
"""
    
    try:
        result = call_llama(prompt)
        if isinstance(result, dict) and "icd_mappings" in result:
            return result["icd_mappings"][:top_k * len(diagnosis_list)]
    except Exception:
        pass
    
    return []