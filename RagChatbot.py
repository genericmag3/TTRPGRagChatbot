from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from streamlit_lottie import st_lottie
import json
import requests
import pandas as pd
import time
import os

import streamlit as st
from streamlit_modal import Modal

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Set up model
model = OllamaLLM(model="phi4:14b")
model.temperature = .6

#Title streamlit chat window
st.title("D&D Q&A Chatbot 🧙‍♂️")

st.info("This app takes your notes from your campaign and passes relevant context from them along with your question to the LLM. It does not store your notes or chat history. Please consult provided references as the AI may hallucinate.")

#Grab custom spinner animation
with open("star-magic.json", "r",errors='ignore') as f:
    magic_spinner = json.load(f)

#Grab custom file upload animation
with open("Magical_Effect_Loading.json", "r",errors='ignore') as f:
    magic_loader = json.load(f)

# Set up retriever in streamlit app
#st.session_state.retriever = retriever

notes_uploaded = False

if "first_chat_key" not in st.session_state:
    st.session_state.first_chat_key = 0

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def update_key():
    st.session_state.uploader_key += 1

note_document = None
databasedir = "./chrome_langchain_db"

# if the database already exists, skip the upload. 
# To do: allow for re-upload of notes via sidebar button
if os.path.isdir(databasedir):
    notes_uploaded = True
    update_key()
    #notes = []
elif st.session_state.uploader_key == 0:
    placeholder = st.empty()
    # Have user upload campaign notes
    with placeholder.container():
        note_document = st.file_uploader("Upload your campaign notes", type=["csv"]) #key=st.session_state.uploader_key, on_change=update_key


#"sentence-transformers/all-MiniLM-L6-v2"

hf_embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu"})
text_splitter = SemanticChunker(hf_embeddings)
vector_store = Chroma(
            collection_name="notes",
            persist_directory=databasedir,
            embedding_function=hf_embeddings
)
retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 9, "score_threshold": 0.1}
)


# Create vector database from file
if note_document is not None:

    #get rid of the file uploader container once file has been selected
    placeholder.empty()

    #start data upload and database creation animation
    animationplaceholder = st.empty()
        # Display the animation initially
    with animationplaceholder.container():
        st_lottie(magic_loader, height=200, key="custom_loading_spinner")
        progress_text = "Casting Vectorization Spell..."
        my_bar = st.progress(0, text=progress_text)


    df = pd.read_csv(note_document)
    #embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db_location = "./chrome_langchain_db"
    percent_complete = 0
    
    if df is not None: # to do: add error checking
        documents = []
        ids = []
        l = 0 #init document ID
        
        for i, row in df.iterrows():
            text = row["Contents"]
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                document = Document(
                    page_content=chunk,
                    metadata={"Title": row["Title"], "Date": row["Date"], "Exerpt Start": chunk[:25], "Exerpt End": chunk[-25:]},
                    id=str(i)
                )
                ids.append(str(l))
                documents.append(document)
                l += 1
            percent_complete = percent_complete + 100/df.shape[0]
            if(percent_complete <= 100):
                my_bar.progress(int(percent_complete), text=progress_text)
        if vector_store != None: # To do: add error checking
            vector_store.add_documents(documents=documents, ids=ids)

        #message_placeholder = st.empty()
        update_key()
        #stop loading animation
        animationplaceholder.empty()
        
        success = st.success("Campaign notes uploaded and processed successfully!")
        notes_uploaded = True
    



# Initialize chat and reference history
if ("messages" not in st.session_state) or ("references" not in st.session_state) or ("buttons" not in st.session_state):
    st.session_state.messages = []
    st.session_state.references = [] # use this to append references per bot response
    st.session_state.buttons = []
    st.session_state.buttoninfo = []
    st.session_state.button_key = 0

k = 0

# st.markdown(  # todo: clean up this custom html. Make it a variable
#                 """
#                     <style>
#                     button {
#                         background: none!important;
#                         border: none;
#                         padding: 0!important;
#                         color: black !important;
#                         text-decoration: none;
#                         cursor: pointer;
#                         border: none !important;
#                     }
#                     button:hover {
#                         text-decoration: none;
#                         color: black !important;
#                     }
#                     button:focus {
#                         outline: none !important;
#                         box-shadow: none !important;
#                         color: black !important;
#                     }
#                     </style>
#                     """,
#                     unsafe_allow_html=True
#                 )

# Display chat messages and references from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        print("message content")
        st.markdown(message["content"])
        print(message["role"])
        if (message["role"] == "assistant"):
            print("creating buttons for the assistant response")
            for refs in st.session_state.references:
                # Create buttons for
                print("generating buttons for a specific bot response")
                #st.session_state.buttons[k:k+len(refs)] 
                for button in st.session_state.buttoninfo[k:k+len(refs)]:
                    #button
                    st.button(button[0], on_click = button[1], args = button[2], key = button[3])
                    #print(button.key)
                k = k + len(refs)
            
            # for item in st.session_state.references[k]:
            #     k = k + 1
            #     if st.button(item.metadata["Date"], key=f"click_{k}"):
            #         st.dialog(str(item.page_content))


# init References history

def reference_button(content):
    modal = Modal(key = "reference_modal", title="Reference Content")
    with modal.container():
        st.markdown(content)
#user_question = # Show chat input at the bottom when a question has been asked.
if notes_uploaded:
    user_question = st.chat_input("Ask a question about the campaign...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question,"avatar":None})
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
        response = chain.invoke({"question": user_question, "notes": notes})  # Pass the query and relevant note documents

        response+="\n______________________________________________________\n"
        response+="Note entry References: \n"
        
        #response+="\n______________________________________________________\n"
        st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
        placeholder.empty()
        with st.chat_message("assistant", avatar="🧙‍♂️"):
            st.markdown(response)    
        st.session_state.references.append(notes) # save group of references per bot response
            # Create a unique button for each reference
        for item in notes:
            st.session_state.buttons.append(st.button(str(item.metadata["Date"]), on_click= reference_button,args=(item.page_content,),  key = f"click_{st.session_state.button_key}"))
            st.session_state.buttoninfo.append([item.metadata["Date"],reference_button, (item.page_content,), f"click_{st.session_state.button_key}"])
            st.session_state.button_key = st.session_state.button_key + 1
        st.session_state.first_chat_key = 1 

#print(st.session_state.references)


#print(st.session_state.references[:len(st.session_state.references[-1])][0].page_content)

#if st.session_state.first_chat_key == 1:  
    #print(st.session_state.references[:len(st.session_state.references[-1])][0].page_content)              
#     #for item in st.session_state.references[-1]:
#         #k = k + 1
#     for i, button in enumerate(st.session_state.buttons[:len(st.session_state.references[-1])]):
#         if button:
#             print(st.session_state.references[:len(st.session_state.references[-1])][i].page_content)
#             #st.dialog(st.session_state.references[:len(st.session_state.references[-1])][i])
