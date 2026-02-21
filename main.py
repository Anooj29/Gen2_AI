# from fastapi import FastAPI
# from pydantic import BaseModel
# import requests

# app = FastAPI()

# class Query(BaseModel):
#     text: str

# @app.post("/generate")
# def generate(query: Query):

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "llama3.1:8b",
#             "prompt": query.text,
#             "stream": False,
#             "temperature": 0.1
#         }
#     )

#     return response.json()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json

from services.text_cleaner import clean_transcript
from services.speaker import separate_speakers
from services.clinical import extract_clinical_data
from services.summary import generate_summary
from services.red_flags import detect_red_flags
from services.icd_mapper import map_icd10
from services.treatment import suggest_treatment_and_medications

app = FastAPI(title="Medical Conversation Analyzer", version="1.0.0")

# Pre-load ICD data on startup for faster first request
@app.on_event("startup")
async def startup_event():
    """Pre-load ICD data on application startup."""
    try:
        from services.icd_mapper import _load_icd_data
        _load_icd_data()
        print("ICD-10 data pre-loaded successfully")
    except Exception as e:
        print(f"Warning: Could not pre-load ICD data: {str(e)}")

@app.get("/")
def root():
    return {"message": "Medical Conversation Analyzer API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

class Segment(BaseModel):
    text: str
    start: Optional[float] = None
    end: Optional[float] = None

class STTInput(BaseModel):
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    transcription: Optional[str] = None
    segments: Optional[List[Dict[str, Any]]] = None

    def get_transcript(self) -> str:
        """Extract transcript from either transcription field or segments."""
        if self.transcription:
            return self.transcription
        elif self.segments:
            return " ".join(seg.get("text", "").strip() for seg in self.segments if isinstance(seg, dict))
        else:
            raise ValueError("Either 'transcription' or 'segments' must be provided")


@app.post("/analyze")
def analyze(data: STTInput):
    try:
        # Step 1: Get transcript from input
        try:
            transcript = data.get_transcript()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail="Empty transcript provided")

        # Step 2: Clean transcript
        cleaned_text = clean_transcript(transcript)

        # Step 3: Separate speakers (use fast heuristic mode by default)
        speaker_data = separate_speakers(cleaned_text, use_llm=False)
        
        if not isinstance(speaker_data, dict):
            return {
                "error": "Speaker separation failed",
                "details": "Invalid return type from speaker separation"
            }

        cleaned_text = speaker_data.get("cleaned_text", "")
        doctor_speech = speaker_data.get("doctor_speech", "")

        if not cleaned_text and not doctor_speech:
            return {
                "error": "No speech detected",
                "details": "Could not extract patient or doctor speech from transcript"
            }

        # Step 4: Red flag detection (early detection for emergencies)
        # Use fast keyword-based mode by default to keep latency low.
        red_flags_result = detect_red_flags(cleaned_text, use_llm=False)
        
        # Step 5: Clinical extraction (only if patient speech exists)
        clinical_json = {}
        if cleaned_text:
            clinical_json = extract_clinical_data(cleaned_text)
            if "error" in clinical_json:
                return {
                    "error": "Clinical extraction failed",
                    "details": clinical_json
                }

        # Step 6: ICD-10 code mapping (if diagnoses are available)
        icd_mappings = []
        if clinical_json and "possible_diagnosis" in clinical_json:
            diagnoses = clinical_json.get("possible_diagnosis", [])
            if isinstance(diagnoses, list) and len(diagnoses) > 0:
                try:
                    # Disable LLM enhancement here for speed; mapping uses CSV similarity.
                    icd_mappings = map_icd10(diagnoses, top_k=3, use_llm_enhancement=False)
                except Exception as e:
                    # Don't fail the entire request if ICD mapping fails
                    icd_mappings = []
                    print(f"Warning: ICD mapping failed: {str(e)}")

        # Step 7: Treatment and medication suggestions
        treatment_suggestions = {}
        if clinical_json and not ("error" in clinical_json):
            try:
                treatment_suggestions = suggest_treatment_and_medications(clinical_json, cleaned_text)
                if "error" in treatment_suggestions:
                    treatment_suggestions = {}
            except Exception as e:
                # Don't fail the entire request if treatment suggestion fails
                treatment_suggestions = {}
                print(f"Warning: Treatment suggestion failed: {str(e)}")

        # Step 8: Summary generation
        summary_json = generate_summary(doctor_speech, cleaned_text)

        if "error" in summary_json:
            return {
                "error": "Summary generation failed",
                "details": summary_json
            }

        # Build comprehensive response
        response = {
            "structured_clinical_data": clinical_json,
            "conversation_summary": summary_json,
            "red_flags": red_flags_result,
            "icd10_mappings": icd_mappings,
            "treatment_suggestions": treatment_suggestions
        }

        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")