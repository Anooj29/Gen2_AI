import re
from typing import Dict, List
from services.llm import call_llama

RED_FLAG_PATTERNS = {
    "cardiac_emergency": ["chest pain", "radiating pain", "sweating with chest pain", "heart attack", "myocardial infarction", "angina"],
    "respiratory_distress": ["shortness of breath", "breathing difficulty", "unable to breathe", "respiratory failure", "choking", "gasping"],
    "neurological_emergency": ["seizure", "unconscious", "stroke", "slurred speech", "loss of consciousness", "fainting", "paralysis"],
    "severe_bleeding": ["severe bleeding", "blood in vomit", "coughing blood", "hemorrhage", "excessive bleeding"],
    "mental_health_crisis": ["suicidal", "self harm", "want to die", "suicide", "harming myself"],
    "severe_pain": ["severe pain", "excruciating pain", "unbearable pain", "10/10 pain"],
    "trauma": ["major trauma", "head injury", "broken bone", "fracture", "accident"]
}

def detect_red_flags(text: str, use_llm: bool = True) -> Dict:
    """
    Detect red flag emergencies in medical conversation.
    Uses hybrid approach: LLM analysis + keyword matching for best accuracy.
    """
    if not text or not text.strip():
        return {
            "red_flag_detected": False,
            "red_flag_categories": [],
            "triage_level": "LOW",
            "urgency": "routine"
        }
    
    # First, do keyword-based detection as quick check
    text_lower = text.lower()
    keyword_categories = []
    
    for category, keywords in RED_FLAG_PATTERNS.items():
        for keyword in keywords:
            if keyword in text_lower:
                keyword_categories.append(category)
                break
    
    # If keywords found or LLM enabled, use LLM for more accurate detection
    if use_llm and (keyword_categories or len(text) > 50):
        try:
            llm_result = _detect_red_flags_with_llm(text)
            if llm_result and "error" not in llm_result:
                # Merge keyword and LLM results
                all_categories = list(set(keyword_categories + llm_result.get("red_flag_categories", [])))
                return {
                    "red_flag_detected": len(all_categories) > 0,
                    "red_flag_categories": all_categories,
                    "triage_level": llm_result.get("triage_level", "HIGH" if all_categories else "LOW"),
                    "urgency": llm_result.get("urgency", "urgent" if all_categories else "routine"),
                    "red_flag_details": llm_result.get("red_flag_details", [])
                }
        except Exception:
            # Fall through to keyword-only if LLM fails
            pass
    
    # Fallback to keyword-based detection
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