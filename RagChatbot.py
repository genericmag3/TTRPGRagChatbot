from langchain_ollama import OllamaLLM
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from streamlit_lottie import st_lottie
import json
import time
import os
import streamlit as st
import numpy as np

#import local modules
import CreateDatabase

# Local Function definitions 

# Set up model
@st.cache_resource
def load_model(modelname):
    model = OllamaLLM(model=modelname)
    return model

def has_subfolders(directory_path):
    if not os.path.isdir(directory_path):
        return False  # Not a valid directory

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path):
            return True
    return False

def create_database_handler(document, databasedir, text_splitter, retriever, vector_store):
    # Start database creation
    gen = CreateDatabase.vectorize_note_document(document, databasedir, text_splitter, retriever, vector_store)
    animationplaceholder = st.empty()
    with animationplaceholder.container():
        st_lottie(magic_loader, height=200, key="custom_loading_spinner")
        progress_text = "Casting Vectorization Spell..."
        vectorization_progress = st.progress(0, text=progress_text)
    returnCode = None
    
     # Use a manual loop to capture the return value from StopIteration
    while True:
        try:
            # Get the next yielded progress value
            progress = next(gen)
            vectorization_progress.progress(progress / 100, text= progress_text + f"{progress:.1f}%")
        except StopIteration as e:
            # This is where your 'return True' or 'return False' from the generator lives
            returnCode = e.value
            animationplaceholder.empty()
            return returnCode

def update_key():
    if st.session_state.uploader_key is not None:
        st.session_state.uploader_key += 1

@st.dialog("Reference Content")
def reference_button(content):
    st.write(content)

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

#Initialize Streamlit chat window 
st.title("D&D Q&A Chatbot 🧙‍♂️")

st.info("This app takes your notes from your campaign and passes relevant context from them along with your question to the local LLM. It does not store your notes or chat history. Please consult provided references as the AI may hallucinate.")

#Grab custom spinner animation
with open("star-magic.json", "r",errors='ignore') as f:
    magic_spinner = json.load(f)

#Grab custom file upload animation
with open("Magical_Effect_Loading.json", "r",errors='ignore') as f:
    magic_loader = json.load(f)

notes_uploaded = False

if ("uploader_key" not in st.session_state) or ("reupload_key" not in st.session_state) or ("model_chosen" not in st.session_state) or ("model_temperature" not in st.session_state):
    st.session_state.uploader_key = 0
    st.session_state.reupload_key = 0
    st.session_state.model_chosen = None 
    st.session_state.model_temperature = None

# Find local ollama models
local_models = ollama.list()
local_model_names = []
for model in local_models.models:
    local_model_names.append(model.model)

# Sidebar model options
sidebar_model_select = st.sidebar.selectbox("Select Model", local_model_names, index = None, placeholder = "Select local LLM...")
sidebar_model_temperature = st.sidebar.selectbox("Select Model Temperature", np.round(np.linspace(0.1, 1.0, 10), 1), index = None, placeholder = "Select local LLM Temperature...")
if ((sidebar_model_select is not None) and (sidebar_model_temperature is not None)):
    st.session_state.model_chosen = sidebar_model_select
    st.session_state.model_temperature = sidebar_model_temperature
    model = load_model(st.session_state.model_chosen)
    model.temperature = st.session_state.model_temperature
else:
    st.session_state.model_chosen = None
    st.session_state.model_temperature = None


note_document = None

databasedir = "./chrome_langchain_db"

# if the database already exists, skip the upload and allow for re-upload sidebar option 
if has_subfolders(databasedir) and st.session_state.reupload_key == False:
    notes_uploaded = True
    update_key()
    sidebar_button = st.sidebar.button('Re-Upload Notes')
    if sidebar_button:
        if os.path.exists(databasedir):
            st.session_state.reupload_key = True
            st.rerun()
elif st.session_state.uploader_key == 0 or st.session_state.reupload_key == True:
    notes_uploaded = False
    placeholder = st.empty()
    # Have user upload campaign notes
    with placeholder.container():
        note_document = st.file_uploader("Upload your campaign notes")

# Init text splitter, retriever, and vector database
text_splitter, retriever, vector_store = CreateDatabase.create_hf_retrival_artifacts(databasedir)

completionmessage = None
# Create vector database from file if file has been uploaded by user
if note_document is not None:
    #get rid of the file uploader container once file has been selected
    st.session_state.reupload_key = 0
    placeholder.empty()
    #start data upload and database creation animation
    notes_uploaded = create_database_handler(note_document, databasedir, text_splitter, retriever, vector_store)
    print(notes_uploaded)
    if(notes_uploaded == True):
        completionmessage = st.empty()
        st.success("Journal processed successfully!")
    else:
        completionmessage = st.empty()
        st.error("Failed to vectorize database. Check file existence or disk space.")

# Initialize session state variables
if ("messages" not in st.session_state) or ("buttoninfo" not in st.session_state) or ("button_key" not in st.session_state) or (st.session_state.reupload_key == True):
    st.session_state.messages = []
    st.session_state.buttoninfo = []
    st.session_state.button_key = 0

i = 0 #  represents index of references, each index can have multiple references and there is one per bot response
# Display chat messages and references from history on app rerun
for message in st.session_state.messages:  
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])
        if (message["role"] == "assistant"):
            if(st.session_state.buttoninfo[i] is not None): 
                for buttoninfo in st.session_state.buttoninfo[i]:
                    st.button(buttoninfo[0], on_click = buttoninfo[1], args = buttoninfo[2], key = buttoninfo[3])
            i = i + 1

if notes_uploaded and (st.session_state.model_chosen is not None) and ((st.session_state.model_temperature is not None)):
    user_question = st.chat_input("Ask a question about the campaign...")
    if user_question:
        if completionmessage is not None:
            completionmessage.empty()
        tempbuttoninfo = []
        st.session_state.messages.append({"role": "user", "content": user_question,"avatar":None})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        placeholder = st.empty()
        # Display the animation initially
        with placeholder.container():
            st_lottie(magic_spinner, height=200, key="custom_spinner")

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
        #references_found = False
        if len(notes) > 0:
            response = chain.invoke({"question": user_question, "notes": notes})  # Pass the query and relevant note documents
            placeholder.empty()
            references_found = True
            with st.chat_message("assistant", avatar="🧙‍♂️"):

            # Only display references if any were found
                if(references_found):
                    response +="\n______________________________________________________\n"
                    response += "Note entry References: \n"
                    st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
                    st.write_stream(stream_data(response))
                    # Create a unique button for each reference
                    for item in notes:
                        tempbuttoninfo.append([item.metadata["Date"],reference_button, (item.page_content,), f"click_{st.session_state.button_key}"])
                        st.button(str(item.metadata["Date"]), on_click= reference_button,args=(item.page_content,),  key = f"click_{st.session_state.button_key}")
                        time.sleep(0.02)
                        # Generate new button key for next button
                        st.session_state.button_key = st.session_state.button_key + 1

                    # Add button information to the session state            
                    st.session_state.buttoninfo.append(tempbuttoninfo) 
        # Save canned response to chat history if no references
        else:
            placeholder.empty()
            response = "Could not find any relevant journal entries for your query. It could be that there is not any relevant information regarding your query in the notes, the question needs to be reworded, or spelling needs to be reviewed."
            st.session_state.buttoninfo.append(None)
            st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
        st.rerun()