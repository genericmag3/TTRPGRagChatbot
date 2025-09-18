from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser
from streamlit_lottie import st_lottie
import json
import requests

#import emoji as moji
import streamlit as st

user_first_interaction = True

# Initialize vector store
vector_store = Chroma(
    collection_name="notes",
    persist_directory="./chrome_langchain_db",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# Set up model
model = OllamaLLM(model="phi4:14b")
model.temperature = .6

#Title streamlit chat window
st.title("D&D Q&A Chatbot 🧙‍♂️")

#Grab custom spinner animation
with open("star-magic.json", "r",errors='ignore') as f:
    magic_spinner = json.load(f)

# Set up retriever in streamlit app
st.session_state.retriever = retriever

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


placeholder = st.empty()

#user_question = # Show chat input at the bottom when a question has been asked.
user_question = st.chat_input("Ask a question about the campaign...")
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    
    placeholder = st.empty()
    # Display the animation initially
    with placeholder.container():
        st_lottie(magic_spinner, height=200, key="custom_spinner")

    #with st.spinner("Consulting texts..."):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful D&D adventure Q&A bot."),
        ("user", "You are an expert in answering questions about a Dungeons and Dragons campaign described in provided documents. The provided documents describe a campaign where the main protagonists are Brocc, Evryn, and Gwendolyn(Gwen). Here are the relevant documents with a date and title from the character Brocc's perspective (sometimes in first person and sometimes in third person): {notes} \n\n Here is the question to answer. Base your answer only off of the provided documents, and no other extraneous material. Do not provide references to the documents.: {question}")
    ])
        
    notes = retriever.invoke(user_question)

    chain = (
        prompt
        | model
        | StrOutputParser()
    )
    response = chain.invoke({"question": user_question, "notes": notes})  # Pass the query as a string, not wrapped in a dictionary

    response+="\n______________________________________________________\n"
    response+="Note entry References(Title, date): \n"
    for item in notes:
        response += "* " + item.metadata["title"] + ", " + item.metadata["date"] + "\n"
        #response += str(item.metadata)
    response+="\n______________________________________________________\n"
    st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
    placeholder.empty()
    with st.chat_message("assistant", avatar="🧙‍♂️"):
        st.markdown(response)
  
