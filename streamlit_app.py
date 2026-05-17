import streamlit as st
from src.app.TTRPGChatBot import TTRPGChatbot
from src.app.CampaignSummarizer import CampaignSummarizer
from src.app.NoteEditor import NoteEditor
from src import app_config

st.set_page_config(page_title="TTRPG Campaign Assistant", page_icon="🧙‍♂️")

def _run_chatbot():
    TTRPGChatbot().run()

def _run_summarizer():
    CampaignSummarizer().run()

def _run_note_editor():
    NoteEditor().run()

chatbot_page = st.Page(_run_chatbot, title="Q&A Chatbot", icon="💬")
summary_page = st.Page(_run_summarizer, title="Campaign Summary", icon="📖")
editor_page = st.Page(_run_note_editor, title="Note Editor", icon="📝")

if app_config.is_remote():
    # Remote (multi-user) build: authenticate before building any page.
    # require_auth() halts the script via st.stop() until the visitor is
    # signed in, so no protected page is ever constructed for a guest.
    from src.auth import gate

    user = gate.require_auth()

    def _run_account():
        gate.render_account_page()

    pages = [chatbot_page, summary_page, editor_page,
             st.Page(_run_account, title="Account", icon="👤")]

    if user.get("is_admin"):
        def _run_admin():
            gate.render_admin_page()
        pages.append(st.Page(_run_admin, title="Admin", icon="🛡️"))

    pg = st.navigation(pages, position="sidebar")
    gate.render_logout_sidebar(user)
    pg.run()
else:
    # Local (single-user) build: unchanged historical behaviour.
    pg = st.navigation([chatbot_page, summary_page, editor_page], position="sidebar")
    pg.run()
