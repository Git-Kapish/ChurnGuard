"""ChurnGuard dashboard entrypoint: `make app` (streamlit run app/streamlit_app.py)."""

import streamlit as st

from views import model, overview, score, simulator

st.set_page_config(page_title="ChurnGuard", layout="wide")

pages = [
    st.Page(
        overview.render,
        title="Overview",
        icon=":material/insights:",
        url_path="overview",
        default=True,
    ),
    st.Page(model.render, title="Model", icon=":material/model_training:", url_path="model"),
    st.Page(
        score.render, title="Score a customer", icon=":material/person_search:", url_path="score"
    ),
    st.Page(
        simulator.render,
        title="Campaign simulator",
        icon=":material/campaign:",
        url_path="simulator",
    ),
]
with st.sidebar:
    st.markdown("### ChurnGuard")
    st.caption("Telecom churn risk, explained, with a budget-aware targeting simulator.")
st.navigation(pages).run()
