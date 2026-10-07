"""
RAG∞ Pro — verified answers from your documents (Streamlit entry point).

    streamlit run app.py

Pages: Chat · Library · Validate · Insights · Settings
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG∞ Pro", page_icon="♾️", layout="wide", initial_sidebar_state="expanded")

from config.settings import get_logger, setup_logging  # noqa: E402
from rag.embeddings import warm_up  # noqa: E402
from ui import pages, state  # noqa: E402
from ui.styles import CSS  # noqa: E402

setup_logging(state.base_settings().log_level)
log = get_logger("app")


@st.cache_resource(show_spinner=False)
def _warm_up() -> bool:
    warm_up(state.base_settings())  # start loading the embedding model in the background
    state.sync_manager()  # start background sync (if Hugging Face storage is configured)
    return True


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.logo("assets/logo.png", size="large", icon_image="assets/icon.png")
    if not state.password_ok():
        return
    _warm_up()
    state.workspace()  # pick up / create the private workspace from the link
    nav = {
        "chat": st.Page(pages.chat_page, title="Chat", icon="💬", url_path="chat", default=True),
        "library": st.Page(pages.library_page, title="Library", icon="📚", url_path="library"),
        "validate": st.Page(pages.validate_page, title="Validate", icon="✅", url_path="validate"),
        "insights": st.Page(pages.insights_page, title="Insights", icon="📊", url_path="insights"),
        "settings": st.Page(pages.settings_page, title="Settings", icon="⚙️", url_path="settings"),
    }
    st.session_state["pages"] = nav
    current = st.navigation(list(nav.values()))
    pages.sidebar()
    current.run()


try:
    main()
except Exception:  # last safety net: log details, show a plain message (never a stack trace)
    log.exception("Unhandled error in the app")
    st.error("Something went wrong. Please try again; details are in the server logs.")
