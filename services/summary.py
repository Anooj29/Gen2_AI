from services.llm import call_llama

def generate_summary(doctor_speech: str, patient_speech: str):
    """
    Generate separate summaries for doctor and patient.
    Returns a dictionary with doctor_summary and patient_summary.
    """
    
    # Combine the conversation for context
    full_conversation = ""
    if doctor_speech:
        full_conversation += f"Doctor: {doctor_speech}\n\n"
    if patient_speech:
        full_conversation += f"Patient: {patient_speech}"

    prompt = f"""
You are a medical documentation assistant. Summarize this doctor-patient conversation.

IMPORTANT RULES:
- Output MUST be valid JSON only.
- Do NOT add explanations, notes, or markdown.
- Start your response with {{ and end with }}.
- Use clear, professional language for doctor summary.
- Use simple, patient-friendly language for patient summary.

Create TWO separate summaries:
1. Doctor Summary: Professional clinical summary for medical records
2. Patient Summary: Simple, easy-to-understand summary for the patient

JSON FORMAT:

{{
  "doctor_summary": "Professional clinical summary here...",
  "patient_summary": "Simple patient-friendly summary here..."
}}

Conversation:
{full_conversation}
"""

    return call_llama(prompt)