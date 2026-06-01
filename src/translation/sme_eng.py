import requests

def translate_to_english(text: str) -> str:
    """
    Translate Northern Sami to English using TartuNLP API.

    Args:
        text: Northern Sami text to translate

    Returns:
        English translation
    """
    url = "https://api.tartunlp.ai/translation/v2"
    body = {
        "text": text,
        "src": "sme",
        "tgt": "eng",
        "domain": "general",
        "application": "OCR Pipeline - Bachelor Thesis"
    }

    try:
        response = requests.post(url, json=body, timeout=10)
        response.raise_for_status()
        return response.json()["result"]
    except requests.exceptions.Timeout:
        return "[Translation timeout - API did not respond]"
    except requests.exceptions.RequestException as e:
        return f"[Translation error: {str(e)}]"
    except (KeyError, ValueError) as e:
        return f"[Translation error: Invalid API response - {str(e)}]"
