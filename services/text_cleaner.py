import re

def clean_transcript(text: str) -> str:
    """
    Clean and normalize transcript text from speech-to-text output.
    Fixes common transcription errors and normalizes formatting.
    """
    if not text:
        return ""
    
    # Common STT corrections
    corrections = {
        "Intimidated Fever": "Intermittent fever",
        "Feral Rite": "febrile episode",
        "short end of breath": "shortness of breath",
        "uh": "",
        "um": "",
        "er": "",
    }
    
    # Apply corrections
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'\.{3,}', '...', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text