import re
import json
from typing import Dict
from services.llm import call_llama


def separate_speakers(text: str, use_llm: bool = True) -> Dict[str, str]:
    """Return a dictionary with keys `patient_speech`, `doctor_speech`, and `cleaned_text`.

    Heuristics:
    - If lines begin with 'Patient:' or 'Doctor:' (case-insensitive), use those labels.
    - Otherwise, use LLM to intelligently separate speakers, or fall back to heuristics.
    - Always include the original cleaned transcript in `cleaned_text` so that
      clinical extraction can see the full context.
    """
    if not text or not text.strip():
        return {"patient_speech": "", "doctor_speech": "", "cleaned_text": ""}

    # Normalize line breaks
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    patient_parts = []
    doctor_parts = []

    # Detect explicit labels
    label_pattern = re.compile(r'^(patient|doctor|dr\.?|physician)\s*[:\-]\s*(.*)$', re.IGNORECASE)
    labeled = False
    for line in lines:
        m = label_pattern.match(line)
        if m:
            labeled = True
            who = m.group(1).lower()
            content = m.group(2).strip()
            # Normalize doctor variations
            if who in ['doctor', 'dr', 'dr.', 'physician']:
                doctor_parts.append(content)
            elif who == 'patient':
                patient_parts.append(content)

    if not labeled and use_llm:
        # Use LLM to separate speakers intelligently
        try:
            llm_result = _separate_with_llm(text)
            if llm_result and "error" not in llm_result:
                # Ensure cleaned_text is always present
                llm_result.setdefault("cleaned_text", text)
                return llm_result
        except Exception:
            # Fall through to heuristic method if LLM fails
            pass

    if not labeled:
        # Fall back to heuristic: split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= 1:
            # Single chunk: assume patient then doctor by splitting in half words
            words = text.split()
            mid = len(words) // 2
            patient_parts.append(' '.join(words[:mid]))
            doctor_parts.append(' '.join(words[mid:]))
        else:
            # Alternate sentences between patient and doctor
            for i, sentence in enumerate(sentences):
                if i % 2 == 0:
                    patient_parts.append(sentence)
                else:
                    doctor_parts.append(sentence)

    patient = "\n".join(p.strip() for p in patient_parts if p)
    doctor = "\n".join(d.strip() for d in doctor_parts if d)

    return {
        "patient_speech": patient,
        "doctor_speech": doctor,
        "cleaned_text": text,
    }


def _separate_with_llm(text: str) -> Dict[str, str]:
    """Use LLM to intelligently separate doctor and patient speech."""
    prompt = f"""
You are analyzing a medical conversation transcript. Separate the speech into doctor and patient parts.

IMPORTANT RULES:
- Output MUST be valid JSON only.
- Do NOT add explanations, notes, or markdown.
- Start your response with {{ and end with }}.
- If you cannot determine who said what, make your best guess based on:
  * Medical terminology and questions suggest doctor
  * Personal symptoms and concerns suggest patient
  * Questions about symptoms suggest doctor
  * Descriptions of personal experience suggest patient

JSON FORMAT:

{{
  "patient_speech": "All patient speech here...",
  "doctor_speech": "All doctor speech here..."
}}

Transcript:
{text}
"""

    result = call_llama(prompt)
    if isinstance(result, dict) and "patient_speech" in result and "doctor_speech" in result:
        return result
    return {}


__all__ = ["separate_speakers"]
