import pandas as pd
import difflib
import os
from pathlib import Path
from typing import List, Dict, Optional
from services.llm import call_llama

# Global variable to store loaded ICD data
icd_data = None
icd_data_loaded = False

def _load_icd_data():
    """Load ICD-10 dataset. Handles path resolution and errors gracefully."""
    global icd_data, icd_data_loaded
    
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
            icd_data["original_desc_col"] = desc_col  # Store for later use
            
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
        
        scored_matches = []
        
        # Search through ICD data
        for _, row in icd_data.iterrows():
            short_desc = str(row.get("short_desc_lower", "")).lower()
            long_desc = str(row.get("long_desc_lower", "")).lower()
            code = str(row.get("code", ""))
            
            # Calculate similarity scores
            score_short = similarity_score(diagnosis_lower, short_desc)
            score_long = similarity_score(diagnosis_lower, long_desc)
            
            # Also check if key terms match
            diagnosis_words = set(diagnosis_lower.split())
            desc_words = set((short_desc + " " + long_desc).split())
            word_overlap = len(diagnosis_words & desc_words) / max(len(diagnosis_words), 1)
            
            # Combined score
            score = max(score_short, score_long) * 0.7 + word_overlap * 0.3
            
            if score > 0.30:  # Lowered threshold for better recall
                # Get description from stored column or try to find it
                desc_col = icd_data.get("original_desc_col", None)
                if desc_col is None:
                    desc_cols = [col for col in icd_data.columns if "description" in col.lower() or "desc" in col.lower()]
                    desc_col = desc_cols[0] if desc_cols else None
                
                description = str(row.get(desc_col, "")) if desc_col else str(row.get("long_desc_lower", ""))
                
                scored_matches.append({
                    "diagnosis": diagnosis_original,
                    "enhanced_diagnosis": enhanced_diagnosis if enhanced_diagnosis else None,
                    "icd_code": code,
                    "description": description,
                    "confidence_score": round(score, 3)
                })
        
        # Sort best matches
        scored_matches = sorted(scored_matches, key=lambda x: x["confidence_score"], reverse=True)
        
        # Deduplicate by ICD code (keep highest confidence)
        seen_codes = {}
        for match in scored_matches:
            code = match["icd_code"]
            if code not in seen_codes or match["confidence_score"] > seen_codes[code]["confidence_score"]:
                seen_codes[code] = match
        
        results.extend(list(seen_codes.values())[:top_k])
    
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