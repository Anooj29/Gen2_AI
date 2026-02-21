from services.llm import call_llama
from typing import Dict, List, Optional


def suggest_treatment_and_medications(clinical_data: Dict, conversation_text: str = "") -> Dict:
    """
    Generate suggested treatment plan and medications based on clinical data.
    
    Args:
        clinical_data: Dictionary containing diagnosis, symptoms, age, gender, etc.
        conversation_text: Optional full conversation text for additional context
    
    Returns:
        Dictionary with suggested_treatment and suggested_medications
    """
    
    # Extract relevant information
    diagnosis = clinical_data.get("possible_diagnosis", [])
    symptoms = clinical_data.get("symptoms_positive", [])
    age = clinical_data.get("age")
    gender = clinical_data.get("gender")
    current_medications = clinical_data.get("medications", [])
    
    # Build context string
    context_parts = []
    if diagnosis:
        context_parts.append(f"Diagnosis: {', '.join(diagnosis)}")
    if symptoms:
        context_parts.append(f"Symptoms: {', '.join(symptoms)}")
    if age:
        context_parts.append(f"Age: {age}")
    if gender:
        context_parts.append(f"Gender: {gender}")
    if current_medications:
        context_parts.append(f"Current medications: {', '.join(current_medications)}")
    
    context = "\n".join(context_parts) if context_parts else "No specific clinical information available."
    
    prompt = f"""
You are a medical treatment recommendation AI. Based on the clinical information provided, suggest appropriate treatment and medications.

IMPORTANT RULES:
- Output MUST be valid JSON only (no extra text, no markdown).
- Do NOT add explanations, comments, or notes.
- Start your response with {{ and end with }}.
- Provide evidence-based, standard medical recommendations.
- Consider patient age and gender when relevant.
- Include dosage, frequency, and duration for medications.
- Suggest medications, prioritizing most important ones.
- Treatment plan should be practical and actionable.

CLINICAL INFORMATION:
{context}

JSON FORMAT:

{{
  "suggested_treatment": [
    "Rest and hydration",
    "Symptomatic management",
    "Follow-up in 3-5 days if symptoms persist"
  ],
  "suggested_medications": [
    {{
      "medication_name": "Paracetamol",
      "dosage": "500 mg",
      "frequency": "Every 6-8 hours",
      "duration": "3-5 days",
      "indication": "For fever and pain relief"
    }},
    {{
      "medication_name": "Ibuprofen",
      "dosage": "400 mg",
      "frequency": "Every 8 hours",
      "duration": "3-5 days",
      "indication": "For anti-inflammatory and pain relief"
    }}
  ],
  "treatment_notes": "Patient should monitor symptoms and seek immediate care if condition worsens."
}}

Generate treatment recommendations:
"""
    
    result = call_llama(prompt)
    
    # Validate and structure the response
    if isinstance(result, dict):
        # Ensure required fields exist
        if "suggested_treatment" not in result:
            result["suggested_treatment"] = []
        if "suggested_medications" not in result:
            result["suggested_medications"] = []
        if "treatment_notes" not in result:
            result["treatment_notes"] = ""
        
        return result
    
    # Fallback if LLM fails
    return {
        "suggested_treatment": [],
        "suggested_medications": [],
        "treatment_notes": "Unable to generate treatment recommendations at this time."
    }

