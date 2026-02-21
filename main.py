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

app = FastAPI(title="Medical Conversation Analyzer", version="1.0.0")

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

        # Step 3: Separate speakers
        speaker_data = separate_speakers(cleaned_text)
        
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

        # Step 4: Clinical extraction (only if patient speech exists)
        clinical_json = {}
        if cleaned_text:
            clinical_json = extract_clinical_data(cleaned_text)
            if "error" in clinical_json:
                return {
                    "error": "Clinical extraction failed",
                    "details": clinical_json
                }

        # Step 5: Summary generation
        summary_json = generate_summary(doctor_speech, cleaned_text)

        if "error" in summary_json:
            return {
                "error": "Summary generation failed",
                "details": summary_json
            }

        return {
            "structured_clinical_data": clinical_json,
            "conversation_summary": summary_json
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")