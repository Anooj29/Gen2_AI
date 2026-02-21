from services.llm import call_llama


def extract_clinical_data(conversation_text: str):
    """
    Extract structured clinical data from the full doctor–patient conversation.
    Returns JSON with age, gender, chief complaint, symptoms, medications, and
    possible diagnoses.
    """

    prompt = f"""
You are a clinical documentation AI. Analyze the following doctor–patient
conversation and extract structured clinical information.

IMPORTANT RULES:
- Output MUST be valid JSON only (no extra text, no markdown).
- Do NOT add explanations, comments, or notes.
- Start your response with {{ and end with }}.
- Base your answers on ANY part of the conversation (doctor or patient).

AGE & GENDER RULES:
- If an exact age is mentioned (e.g. "45-year-old male"), set "age" to that number.
- If age is not clearly mentioned, leave "age" as null (do NOT guess a number).
- If gender is clearly mentioned or implied (man, woman, male, female, boy, girl),
  set "gender" to "male" or "female". Otherwise leave "gender" as null.

MEDICATION RULES:
- "medications" must be a list of strings.
- EACH string MUST include the full phrase: drug name + dose + frequency,
  if they are present in the conversation.
- Do NOT shorten to just the drug name.
- Example item: "Paracetamol 500 mg twice daily".

DIAGNOSIS RULES:
- "possible_diagnosis" must be a list of 1–3 short clinical diagnoses if there
  is enough information (e.g. "Acute viral upper respiratory infection").
- Only leave "possible_diagnosis": [] if there is truly no clinical information.

JSON FORMAT (schema only, values are examples/placeholders):

{{
  "age": 45,
  "gender": "male",
  "chief_complaint": "Fever and cough for 3 days",
  "symptoms_positive": ["fever", "cough", "body ache"],
  "symptoms_negative": ["no shortness of breath", "no chest pain"],
  "duration": "3 days",
  "medications": ["Paracetamol 500 mg twice daily"],
  "possible_diagnosis": ["Acute viral upper respiratory infection"]
}}

Conversation (doctor and patient):
{conversation_text}
"""

    return call_llama(prompt)