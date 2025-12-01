"""Configuration for automated EDA app."""

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
MAX_ROWS_FOR_FULL_ANALYSIS = 100_000

LLM_BACKENDS = {
    "dummy": {
        "type": "dummy",
        "description": "Built-in dummy backend that formats a report without calling any external model.",
    },
    "ollama": {
        "type": "ollama",
        "description": "Skeleton config for a local Ollama server (e.g., Llama3, Mistral).",
        "base_url_env": "OLLAMA_BASE_URL",
        "model": "llama3:8b",
    },
    "hf_inference": {
        "type": "huggingface_inference",
        "description": "Skeleton config for Hugging Face Inference API (e.g., Qwen2.5-7B).",
        "api_url_env": "HF_API_URL",
        "api_key_env": "HF_API_TOKEN",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
    },
}

DEFAULT_LLM_BACKEND = "dummy"
