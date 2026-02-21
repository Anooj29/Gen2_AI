import re
from typing import Dict, List
from services.llm import call_llama

# More specific patterns - only flag TRUE emergencies
RED_FLAG_PATTERNS = {
    "cardiac_emergency": [
        "heart attack", "myocardial infarction", "cardiac arrest", 
        "chest pain radiating", "crushing chest pain", "sweating with chest pain"
    ],
    "respiratory_distress": [
        "unable to breathe", "respiratory failure", "choking", "gasping for air",
        "severe shortness of breath", "cannot breathe"
    ],
    "neurological_emergency": [
        "seizure", "unconscious", "stroke", "loss of consciousness", 
        "paralysis", "sudden weakness", "cannot move"
    ],
    "severe_bleeding": [
        "severe bleeding", "excessive bleeding", "hemorrhage", 
        "blood in vomit", "coughing up blood", "profuse bleeding"
    ],
    "mental_health_crisis": [
        "suicidal", "want to die", "suicide", "harming myself", 
        "planning to hurt myself"
    ],
    "severe_pain": [
        "excruciating pain", "unbearable pain", "10/10 pain", 
        "worst pain ever", "cannot tolerate pain"
    ],
    "trauma": [
        "major trauma", "severe head injury", "severe head injury",
        "multiple fractures", "severe accident"
    ]
}

# Exclude these common phrases that are NOT emergencies
NON_EMERGENCY_PATTERNS = [
    "mild", "slight", "a little", "some", "minor", "low grade fever",
    "simple", "routine", "normal", "just", "only", "slight pain",
    "mild discomfort", "manageable", "bearable"
]

def detect_red_flags(text: str, use_llm: bool = False) -> Dict:
    """
    Detect red flag emergencies in medical conversation.
    Uses conservative keyword matching - only flags TRUE emergencies.
    """
    if not text or not text.strip():
        return {
            "red_flag_detected": False,
            "red_flag_categories": [],
            "triage_level": "LOW",
            "urgency": "routine"
        }
    
    text_lower = text.lower()
    
    # First check: Exclude if text contains non-emergency qualifiers
    has_non_emergency = any(pattern in text_lower for pattern in NON_EMERGENCY_PATTERNS)
    
    # Second check: Look for specific emergency keywords
    keyword_categories = []
    
    for category, keywords in RED_FLAG_PATTERNS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Additional context check: make sure it's not negated or mild
                keyword_index = text_lower.find(keyword)
                # Check surrounding context (50 chars before/after)
                context_start = max(0, keyword_index - 50)
                context_end = min(len(text_lower), keyword_index + len(keyword) + 50)
                context = text_lower[context_start:context_end]
                
                # Skip if context suggests it's NOT an emergency
                if any(neg in context for neg in ["no ", "not ", "mild ", "slight ", "minor ", "a little "]):
                    continue
                    
                keyword_categories.append(category)
                break
    
    # Only flag if we found emergency keywords AND no non-emergency qualifiers
    # OR if it's a very clear emergency (like "heart attack")
    clear_emergencies = ["heart attack", "cardiac arrest", "stroke", "seizure", "unconscious", 
                         "respiratory failure", "suicidal", "severe bleeding"]
    is_clear_emergency = any(emergency in text_lower for emergency in clear_emergencies)
    
    if has_non_emergency and not is_clear_emergency:
        # Likely not an emergency if it has non-emergency qualifiers
        keyword_categories = []
    
    # If keywords found and LLM enabled, use LLM for validation
    if use_llm and keyword_categories:
        try:
            llm_result = _detect_red_flags_with_llm(text)
            if llm_result and "error" not in llm_result:
                # Use LLM result if it confirms, otherwise be conservative
                if llm_result.get("red_flag_detected", False):
                    return {
                        "red_flag_detected": True,
                        "red_flag_categories": llm_result.get("red_flag_categories", keyword_categories),
                        "triage_level": llm_result.get("triage_level", "HIGH"),
                        "urgency": llm_result.get("urgency", "urgent"),
                        "red_flag_details": llm_result.get("red_flag_details", [])
                    }
        except Exception:
            pass
    
    # Return conservative result
    return {
        "red_flag_detected": len(keyword_categories) > 0,
        "red_flag_categories": keyword_categories,
        "triage_level": "HIGH" if keyword_categories else "LOW",
        "urgency": "urgent" if keyword_categories else "routine"
    }


def _detect_red_flags_with_llm(text: str) -> Dict:
    """Use LLM to intelligently detect red flag emergencies."""
    prompt = f"""
You are a medical triage AI. Analyze this doctor-patient conversation for red flag emergencies that require immediate attention.

RED FLAG CATEGORIES:
- cardiac_emergency: Chest pain, heart attack, cardiac symptoms
- respiratory_distress: Breathing difficulties, respiratory failure
- neurological_emergency: Seizures, stroke, loss of consciousness, paralysis
- severe_bleeding: Excessive bleeding, hemorrhage
- mental_health_crisis: Suicidal ideation, self-harm
- severe_pain: Unbearable or severe pain (8+/10)
- trauma: Major injuries, fractures, head trauma

IMPORTANT RULES:
- Output MUST be valid JSON only (no extra text, no markdown).
- Do NOT add explanations, comments, or notes.
- Start your response with {{ and end with }}.
- Only flag TRUE emergencies, not routine symptoms.
- Be conservative - false positives are better than missing emergencies.

JSON FORMAT:

{{
  "red_flag_detected": true/false,
  "red_flag_categories": ["cardiac_emergency", "respiratory_distress", ...],
  "triage_level": "HIGH" or "MEDIUM" or "LOW",
  "urgency": "critical" or "urgent" or "routine",
  "red_flag_details": [
    {{
      "category": "cardiac_emergency",
      "description": "Patient reports chest pain radiating to left arm",
      "severity": "high"
    }}
  ]
}}

Conversation:
{text}
"""
    
    result = call_llama(prompt)
    if isinstance(result, dict) and "red_flag_detected" in result:
        return result
    return {}