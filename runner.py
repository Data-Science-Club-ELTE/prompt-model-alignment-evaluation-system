import requests

OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120


def generate(prompt: str, model: str = "llama3.1:8b") -> str:
    """Send a prompt to Ollama and return the output text."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def embed(text: str, model: str = "nomic-embed-text") -> list:
    """Get an embedding vector for a piece of text."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def run_multiple(prompt: str, model: str, n: int = 3) -> list:
    """Run the same prompt n times and return all outputs."""
    print(f"  Running prompt {n} times for consistency check...")
    outputs = []
    for i in range(n):
        try:
            outputs.append(generate(prompt, model))
        except Exception as e:
            print(f"  Warning: run {i+1} failed ({e})")
    return outputs