import streamlit as st
from src.app.TTRPGChatBot import TTRPGChatbot
from src.app.CampaignSummarizer import CampaignSummarizer
from src.app.NoteEditor import NoteEditor
from src.utils.NavigationHandler import disable_navigation_if_processing

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

pg = st.navigation([chatbot_page, summary_page, editor_page], position="sidebar")
disable_navigation_if_processing()
pg.run()
