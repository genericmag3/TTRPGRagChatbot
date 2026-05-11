import streamlit as st
import src.app.TTRPGChatBot as TTRPGChatBotModule

st.set_page_config(page_title="TTRPG Campaign Assistant", page_icon="🧙‍♂️")

def _run_chatbot():
    chatbot = TTRPGChatBotModule.TTRPGChatbot()
    chatbot.run()

chatbot_page = st.Page(_run_chatbot, title="Q&A Chatbot", icon="💬")
summary_page = st.Page("pages/1_Campaign_Summary.py", title="Campaign Summary", icon="📖")

# Refresh every run so page objects stay current for cross-page links
st.session_state._chatbot_page = chatbot_page

pg = st.navigation([chatbot_page, summary_page], position="hidden")
pg.run()
