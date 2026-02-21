import requests
import re
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

# Global LLM options tuned for latency vs quality
LLM_OPTIONS = {
    # Limit how many tokens the model can generate to keep responses fast.
    # JSON outputs for our use-cases are short, so 512 is plenty.
    "num_predict": 512,
    # You can also tune these further if needed:
    # "num_ctx": 2048,
}


def call_llama(prompt: str, expect_json: bool = True):
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1,
        }

        # Attach options if defined (helps control latency)
        if LLM_OPTIONS:
            payload["options"] = LLM_OPTIONS

        response = requests.post(OLLAMA_URL, json=payload, timeout=120)

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