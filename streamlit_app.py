import streamlit as st
import src.app.TTRPGChatBot as TTRPGChatBotModule
from src.utils.LLMHandler import LLMHandler
from src.utils.SummaryHandler import SummaryHandler
from src.app.CampaignSummarizer import CampaignSummarizer

st.set_page_config(page_title="TTRPG Campaign Assistant", page_icon="🧙‍♂️")

def _run_chatbot():
    chatbot = TTRPGChatBotModule.TTRPGChatbot()
    chatbot.run()

def _run_summarizer():
    llm_handler = LLMHandler()
    summary_handler = SummaryHandler(llm_handler)
    CampaignSummarizer(llm_handler, summary_handler).run()

chatbot_page = st.Page(_run_chatbot, title="Q&A Chatbot", icon="💬")
summary_page = st.Page(_run_summarizer, title="Campaign Summary", icon="📖")

pg = st.navigation([chatbot_page, summary_page], position="sidebar")
pg.run()
