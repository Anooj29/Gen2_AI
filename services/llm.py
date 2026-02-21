import requests
import re
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

def call_llama(prompt: str, expect_json: bool = True):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            },
            timeout=120
        )

        response.raise_for_status()

        raw_output = response.json().get("response", "").strip()

        if not expect_json:
            return raw_output

        # 🔥 Extract JSON block safely
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)

        if not json_match:
            return {
                "error": "No valid JSON found in model output",
                "raw_output": raw_output
            }

        json_text = json_match.group()

        try:
            parsed_json = json.loads(json_text)
            return parsed_json
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON format from model",
                "raw_output": raw_output
            }

    except requests.exceptions.RequestException as e:
        return {
            "error": "Failed to connect to LLM",
            "details": str(e)
        }