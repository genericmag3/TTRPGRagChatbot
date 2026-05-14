import streamlit as st
from src.app.TTRPGChatBot import TTRPGChatbot
from src.app.CampaignSummarizer import CampaignSummarizer

st.set_page_config(page_title="TTRPG Campaign Assistant", page_icon="🧙‍♂️")

def _run_chatbot():
    TTRPGChatbot().run()

def _run_summarizer():
    CampaignSummarizer().run()

chatbot_page = st.Page(_run_chatbot, title="Q&A Chatbot", icon="💬")
summary_page = st.Page(_run_summarizer, title="Campaign Summary", icon="📖")

pg = st.navigation([chatbot_page, summary_page], position="sidebar")
pg.run()
