"""Streamlit UI for automated EDA web app."""

from __future__ import annotations

import traceback
from typing import Optional

import pandas as pd
import streamlit as st

from . import analysis_engine, config, llm_reporter


def _load_dataframe(uploaded_file) -> Optional[pd.DataFrame]:
    """Load CSV or Excel file into a DataFrame."""

    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"파일을 불러오는 중 오류가 발생했습니다: {exc}")
        return None


def main() -> None:
    """Render the Streamlit UI."""

    st.set_page_config(page_title="자동 EDA", layout="wide")
    st.title("자동화된 EDA 도구")
    st.write(
        "업로드된 CSV/Excel 데이터셋에 대해 자동으로 EDA를 수행하고, \n"
        "간단한 베이스라인 모델 결과와 요약 리포트를 제공합니다. \n"
        "본 도구는 의사결정을 보조하기 위한 참고 정보이며, 최종 결정 시스템이 아닙니다."
    )

    uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요", type=["csv", "xlsx", "xls"])

    target_column = st.text_input("타겟 컬럼 이름 (선택사항)", value="")

    backend_options = list(config.LLM_BACKENDS.keys())
    backend_labels = [
        f"{name} - {config.LLM_BACKENDS[name].get('description', '')}" for name in backend_options
    ]
    selected_label = st.selectbox("LLM/SLM 백엔드 선택", backend_labels)
    selected_backend = backend_options[backend_labels.index(selected_label)]

    if st.button("분석 실행"):
        if uploaded_file is None:
            st.error("먼저 파일을 업로드해 주세요.")
            return

        df = _load_dataframe(uploaded_file)
        if df is None or df.empty:
            st.error("데이터 로드에 실패했거나 빈 데이터입니다.")
            return

        st.subheader("데이터 미리보기")
        st.dataframe(df.head())

        explicit_target = target_column.strip() or None
        try:
            analysis_result = analysis_engine.run_analysis(df, explicit_target=explicit_target)
        except Exception:
            st.error("분석 중 오류가 발생했습니다. 자세한 내용은 로그를 확인하세요.")
            traceback.print_exc()
            return

        summary = analysis_result.get("analysis_summary", {})
        overview = summary.get("dataset_overview", {})
        st.subheader("데이터셋 개요")
        st.json(overview)

        st.subheader("컬럼 요약")
        st.json(summary.get("column_summaries", {}))

        st.subheader("문제 유형")
        st.json(analysis_result.get("problem_type"))

        if summary.get("baseline_results"):
            st.subheader("베이스라인 결과")
            st.json(summary.get("baseline_results"))

        report_text = llm_reporter.generate_report(
            analysis_result=analysis_result,
            language="ko",
            backend_name=selected_backend,
            backend_config=config.LLM_BACKENDS.get(selected_backend, {}),
        )

        st.subheader("자동 생성 리포트")
        st.markdown(report_text)


if __name__ == "__main__":
    main()
