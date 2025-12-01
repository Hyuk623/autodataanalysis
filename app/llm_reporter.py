"""LLM/SLM backend abstraction and report generation."""

from __future__ import annotations

from typing import Any, Dict, Protocol

from . import config


class BaseLLMClient(Protocol):
    """Base protocol for LLM clients."""

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - interface
        ...


class DummyLLMClient:
    """Dummy backend that simply formats and echoes the prompt."""

    def __init__(self, backend_name: str = "dummy", model_name: str | None = None):
        self.backend_name = backend_name
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return "DUMMY LLM BACKEND:\n" + prompt


class OllamaLLMClient:
    """Skeleton client for a local Ollama server."""

    def __init__(self, base_url: str, model: str):
        """Prepare a client for a local Ollama deployment."""

        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - skeleton
        raise NotImplementedError("OllamaLLMClient is not implemented yet.")


class HFInferenceLLMClient:
    """Skeleton client for Hugging Face Inference API."""

    def __init__(self, api_url: str, api_key: str, model_id: str):
        """Prepare a client for Hugging Face Inference endpoints."""

        self.api_url = api_url
        self.api_key = api_key
        self.model_id = model_id

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - skeleton
        raise NotImplementedError("HFInferenceLLMClient is not implemented yet.")


def get_llm_client(backend_name: str, backend_config: Dict[str, Any]) -> BaseLLMClient:
    """Return an LLM client instance for the requested backend."""

    backend_type = backend_config.get("type", backend_name)

    if backend_type == "ollama":
        base_url = backend_config.get("base_url") or backend_config.get("base_url_env")
        model = backend_config.get("model")
        return OllamaLLMClient(base_url=base_url or "http://localhost:11434", model=model or "llama3")
    if backend_type == "huggingface_inference":
        api_url = backend_config.get("api_url") or backend_config.get("api_url_env")
        api_key = backend_config.get("api_key") or backend_config.get("api_key_env")
        model_id = backend_config.get("model_id", "")
        return HFInferenceLLMClient(api_url=api_url or "", api_key=api_key or "", model_id=model_id)

    return DummyLLMClient(backend_name=backend_name, model_name=backend_config.get("model"))


def _build_prompt(analysis_result: dict, language: str = "ko") -> str:
    """Build a prompt string from the analysis result."""

    overview = analysis_result.get("analysis_summary", {}).get("dataset_overview", {})
    problem_info = analysis_result.get("problem_type", {})
    target_info = analysis_result.get("target_info", {})
    quality_issues = analysis_result.get("analysis_summary", {}).get("quality_issues", [])
    baseline = analysis_result.get("analysis_summary", {}).get("baseline_results")

    lines = [
        "데이터셋 자동 EDA 결과를 한국어로 요약해 주세요.",
        "주어진 정보만 사용하고 새로운 수치를 만들지 마세요.",
    ]

    lines.append(f"- 행/열 수: {overview.get('n_rows')} / {overview.get('n_cols')}")
    lines.append(f"- 문제 유형: {problem_info.get('type')} (이유: {problem_info.get('reason')})")
    lines.append(f"- 타겟 컬럼: {target_info.get('target_column')} (이유: {target_info.get('reason')})")

    if quality_issues:
        lines.append("- 데이터 품질 이슈:")
        for issue in quality_issues:
            lines.append(f"  * {issue}")

    if baseline:
        lines.append("- 베이스라인 모델 결과:")
        for k, v in baseline.items():
            lines.append(f"  * {k}: {v}")

    lines.append("가능한 한 간결하게 주요 통찰을 bullet point로 정리해 주세요.")
    return "\n".join(lines)


def generate_report(
    analysis_result: dict,
    language: str = "ko",
    backend_name: str = "dummy",
    backend_config: dict | None = None,
) -> str:
    """Generate a human-readable EDA report via the selected backend."""

    backend_config = backend_config or config.LLM_BACKENDS.get(backend_name, {})
    prompt = _build_prompt(analysis_result, language=language)

    if backend_name == "dummy" or backend_config.get("type") == "dummy":
        # Return formatted prompt without calling external services.
        return "\n".join(
            [
                "# 자동 EDA 요약 (Dummy Backend)",
                prompt,
            ]
        )

    client = get_llm_client(backend_name, backend_config)
    return client.generate(prompt)

