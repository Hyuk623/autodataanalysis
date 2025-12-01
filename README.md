# Automated EDA Web App

This project provides a small, production-ready skeleton for an automated EDA (Exploratory Data Analysis) web application built with **Streamlit**. Users can upload CSV or Excel files, run quick EDA with baseline models, and generate a Korean report via a configurable LLM/SLM backend. The current implementation includes a dummy backend; placeholders are provided for future open-source model integrations.

## Architecture
- **app/analysis_engine.py**: Performs column type inference, target inference, problem type detection, EDA, and baseline modeling (classification, regression, time series, unsupervised clustering).
- **app/llm_reporter.py**: Abstracts LLM/SLM backends, offering a dummy backend and skeletons for Ollama and Hugging Face Inference.
- **app/app_ui.py**: Streamlit interface for uploading data, selecting target/backends, running analysis, and viewing results and reports.
- **app/config.py**: Central configuration for defaults and available LLM/SLM backends.
- **tests/**: Basic pytest coverage for key analysis functions.
- **notebooks/examples.ipynb**: Minimal notebook showing how to use the analysis engine directly.

## Installation
```bash
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run app/app_ui.py
```

## Switching LLM/SLM Backends
- Edit `config.LLM_BACKENDS` and `config.DEFAULT_LLM_BACKEND` to add or change backends.
- Choose the backend from the dropdown in the Streamlit UI.
- The current version uses a dummy backend; external API calls are not implemented yet.

## Notes
- All statistical computations occur in Python. The LLM/SLM layer only formats the provided summary into a human-readable report.
- Replace the dummy backend implementations with real open-source LLM/SLM clients (e.g., Llama3 via Ollama, Hugging Face Inference) as needed.

---

## 한국어 안내
업로드된 CSV/Excel 데이터를 자동으로 분석하고 간단한 베이스라인 모델 결과 및 요약 리포트를 제공합니다. 현재는 더미 LLM 백엔드만 포함되어 있으며, 추후 오픈소스 모델 연동을 위해 스켈레톤 코드가 준비되어 있습니다.
