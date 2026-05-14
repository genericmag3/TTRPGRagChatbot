import streamlit as st
from streamlit_lottie import st_lottie
from langchain_core.prompts import ChatPromptTemplate
import json
import time
import os
import numpy as np
import uuid

#import local modules
from ..utils import DatabaseHandler
from ..utils import LLMHandler
from ..utils import SummaryHandler

class TTRPGChatbot:
    _USERDATAFILE = "data//user_data.json"

    def __init__(self):
        # constant class variables
        self._DATABASEDIR = DatabaseHandler.DATABASE_DIR
        self._PROMPTEMPLATE = ChatPromptTemplate.from_messages([
                                ("system", "You are an expert in answering questions about a TTRPG campaign described in provided documents. "
                                "The provided documents describe a campaign where the party members (player characters) are {partymembers}. "
                                "Here are the relevant documents from {notetaker}'s perspective (could be in first person or third person): {notes}"
                                "/n Base your answers only off of the provided documents. Do not use extranious information to answer the question. Do not provide references to the documents."
                                ),
                                ("user", "{question}"
                                )
                            ])

        # init class datamembers
        if 'databasehandler' not in st.session_state:
            st.session_state.databasehandler = DatabaseHandler.DatabaseHandler()
        self.databasehandler = st.session_state.databasehandler # Class should handle all interactions with the vector database, including creation, retrieval, and updates. Vector store instance should exist within this class.
        self.llmhandler = LLMHandler.LLMHandler() # Class should handle model loading and inference. Model instance should exist within this class.
        self.summaryhandler = SummaryHandler.SummaryHandler(self.llmhandler)

        # init static chat window UI
        self.__init_UI()

        # init state variables and handle any errors with user data file
        if (self.__init_state_variables() == False):
            st.rerun() # retry initialization after handling error

    def __init_state_variables(self):
        # Initialize session state variables for model, database upload handling, and document retriever for storing in memory to avoid reload
        if ("reupload_key" not in st.session_state) or ("model_name" not in st.session_state) or ("model_temperature" not in st.session_state) or ("notes_uploaded" not in st.session_state) or ("messages" not in st.session_state) or ("buttoninfo" not in st.session_state) or ("button_key" not in st.session_state) or ("party_members" not in st.session_state) or ("delete_index" not in st.session_state) or ("summary_generated" not in st.session_state):
            if os.path.isfile(self._USERDATAFILE):
                try:
                    with open(self._USERDATAFILE, "r") as f:
                        user_data = json.load(f)
                except Exception as e:
                    st.error(f"Error loading user data: {e}")
                    os.remove(self._USERDATAFILE)
                    return False
                st.session_state.reupload_key = 0
                st.session_state.model_name = user_data.get("model_name")
                st.session_state.model_temperature = user_data.get("model_temperature")
                st.session_state.notes_uploaded = user_data.get("notes_uploaded")
                st.session_state.messages = []
                st.session_state.buttoninfo = []
                st.session_state.button_key = 0
                st.session_state.party_members = user_data.get("party_members")
                st.session_state.delete_index = None
                st.session_state.summary_generated = self.summaryhandler.summary_exists()
                if st.session_state.model_name is not None and st.session_state.model_temperature:
                    self.llmhandler.load_model(str(st.session_state.model_name), st.session_state.model_temperature)
            # 1st run or missing user options data file, initialize session state variables to default values
            else:
                st.session_state.reupload_key = 0
                st.session_state.model_name = None
                st.session_state.model_temperature = None
                st.session_state.notes_uploaded = False
                st.session_state.messages = []
                st.session_state.buttoninfo = []
                st.session_state.button_key = 0
                st.session_state.party_members = [{'id': str(uuid.uuid4()), 'name': "", 'note_taker': False}]
                st.session_state.delete_index = None
                st.session_state.summary_generated = self.summaryhandler.summary_exists()

        return True

    def __process_model_options(self):
        # Find local ollama models
        local_model_names = [model.model for model in self.llmhandler.get_available_models()]
        temperature_options = np.round(np.linspace(0.1, 1.0, 10), 1)
        # Generate sidebar options
        with st.sidebar:
            st.header("🔧 Model Options")
            sidebar_model_select = st.sidebar.selectbox("Select Model", local_model_names,
                                                        placeholder="Select local LLM...",
                                                        index=local_model_names.index(st.session_state.model_name) if st.session_state.model_name in local_model_names else None)
            sidebar_model_temperature = st.sidebar.slider("Select Model Temperature",
                                                        min_value=0.0,
                                                        max_value=1.0,
                                                        value=st.session_state.model_temperature if st.session_state.model_temperature is not None else 0.7,
                                                        step=0.1,
                                                        key="model_temperature_slider")
            if ((sidebar_model_select is not None) and (sidebar_model_temperature is not None)):
                st.session_state.model_name = sidebar_model_select
                st.session_state.model_temperature = sidebar_model_temperature
                self.llmhandler.load_model(sidebar_model_select, sidebar_model_temperature)
                self.__save_user_data()
            else:
                st.session_state.model_name = None
                st.session_state.model_temperature = None

    def __save_user_data(self):
        existing = {}
        if os.path.isfile(self._USERDATAFILE):
            try:
                with open(self._USERDATAFILE, "r") as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing.update({
            "model_name": st.session_state.model_name,
            "model_temperature": st.session_state.model_temperature,
            "notes_uploaded": st.session_state.notes_uploaded,
            "party_members": st.session_state.party_members,
        })
        with open(self._USERDATAFILE, "w") as f:
            json.dump(existing, f)

    def __process_journal_options(self):
        with st.sidebar:
            st.header("📜🪶 Journal Options")

            #check if any note taker is already selected
            any_note_taker_selected = any(member.get('note_taker', False) for member in st.session_state.party_members)
            # Iterate over the list party members
            for i, member in enumerate(st.session_state.party_members):
                m_id = member['id']

                # Name input and delete button columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    new_name = st.text_input(
                        f"Member {i+1}",
                        key=f"input_{m_id}",
                        value=member['name'],
                        label_visibility="collapsed"  # Makes it more compact
                    )

                with col2:
                    is_disabled = any_note_taker_selected and not member.get('note_taker', False)
                    st.checkbox(f"Note Taker", key=f"note_taker_{m_id}", help="Check if this party member is the note taker for processed notes.", on_change=self.__toggle_note_taker, args=(m_id,), disabled=is_disabled, value=member.get('note_taker', True))
                    # Auto-update name when changed
                    if new_name != member['name']:
                        member['name'] = new_name.strip()
                        st.rerun()

                with col3:
                    st.button(
                        "🗑️",
                        key=f"delete_{m_id}",
                        use_container_width=False,
                        type="secondary",  # Smaller secondary button style
                        on_click=self.__delete_member,
                        args=(m_id,)
                    )

            if st.button('➕ Add New Member', type='primary'):
                st.session_state.party_members.append({'id': str(uuid.uuid4()), 'name': None, 'note_taker': False})
                st.rerun()

        note_document = None
        model_ready = st.session_state.model_name is not None

        if self.summaryhandler.raw_notes_exist() and (st.session_state.reupload_key == False):
            st.session_state.notes_uploaded = True
            sidebar_button = st.sidebar.button('Re-Upload Notes', disabled=not model_ready,
                                               help="Select a model before re-uploading notes." if not model_ready else None)
            if sidebar_button:
                self.databasehandler.clear_database(self._DATABASEDIR)
                for stale in ("data/raw_notes.json", "data/campaign_summary.json"):
                    if os.path.isfile(stale):
                        os.remove(stale)
                st.session_state.summary_generated = False
                st.session_state.reupload_key = True
                self.__reset_chat_history()
                st.rerun()
        # if the database does not exist, or the user opted to re-upload notes, have user upload notes and create database
        else:
            st.session_state.notes_uploaded = False
            placeholder = st.empty()
            with placeholder.container():
                if not model_ready:
                    st.info("⚠️ Please select a model in **Model Options** before uploading campaign notes.")
                else:
                    note_document = st.file_uploader("Upload your campaign notes", type=["txt", "docx", "csv"])
        # Init text splitter, retriever, and vector database
        self.databasehandler.create_retrival_artifacts(self._DATABASEDIR)
        # Check to see if user uploaded notes
        if note_document is not None:
            placeholder.empty()
            st.session_state.reupload_key = False
            st.session_state.notes_uploaded = self.__create_database_handler(note_document)
            if st.session_state.notes_uploaded:
                with st.toast("📜🪶 Notes processed successfully!", icon="🧙‍♂️"):
                    pass
                st.toast("📖 A campaign summary is now available on the Campaign Summary page!", icon="📖")
            else:
                with st.toast("❌ Notes processing failed! Check disk space or existence of journal.", icon="🧙‍♂️"):
                    pass

        self.__save_user_data()

    def __create_database_handler(self, document):
        gen = self.databasehandler.generate_database(document, self._DATABASEDIR)

        with open("assets/Magical_Effect_Loading.json", "r", errors='ignore') as f:
            magic_loader = json.load(f)
        animationplaceholder = st.empty()
        with animationplaceholder.container():
            st_lottie(magic_loader, height=200, key="custom_loading_spinner")
            progress_text = "Casting Vectorization Spell..."
            vectorization_progress = st.progress(0, text=progress_text)
        returnCode = None

        while True:
            try:
                progress = next(gen)
                vectorization_progress.progress(progress / 100, text=progress_text + f" {progress:.1f}%")
            except StopIteration as e:
                returnCode = e.value
                animationplaceholder.empty()
                break

        if not returnCode:
            return returnCode

        # Persist the raw notes DataFrame so the summary page and SummaryHandler can access it
        if self.databasehandler.last_processed_df is not None:
            os.makedirs("data", exist_ok=True)
            self.databasehandler.last_processed_df.to_json("data/raw_notes.json")

        return returnCode


    def __reset_chat_history(self):
        st.session_state.messages = []
        st.session_state.buttoninfo = []
        st.session_state.button_key = 0

    def __update_message_history(self):
        i = 0 #  represents index of references, each index can have multiple references and there is one per bot response
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                st.markdown(message["content"])
                if (message["role"] == "assistant"):
                    if(st.session_state.buttoninfo[i] is not None):
                        for buttoninfo in st.session_state.buttoninfo[i]:
                            st.button(buttoninfo[0], on_click = buttoninfo[1], args = buttoninfo[2], key = buttoninfo[3])
                    i = i + 1

    def __process_chat(self):
        if st.session_state.notes_uploaded and (st.session_state.model_name is not None):
            user_question = st.chat_input("Ask a question about the campaign...")
            if user_question:
                tempbuttoninfo = []
                st.session_state.messages.append({"role": "user", "content": user_question,"avatar":None})
                with st.chat_message("user"):
                    st.markdown(user_question)

                placeholder = st.empty()

                # Load animation from json
                with open("assets/star-magic.json", "r",errors='ignore') as f:
                    magic_spinner = json.load(f)

                # Display the animation initially
                with placeholder.container():
                    st_lottie(magic_spinner, height=200, key="custom_spinner")

                # retrieve relevant notes from the datababse based on the user query
                notes = self.databasehandler.retrieve_notes(user_question)

                # Pass user query plus relevant notes to the model and get response if relevant notes are found
                if len(notes) > 0:
                    members = [member['name'] for member in st.session_state.party_members]
                    if len(members) > 1:
                        formatted_members = ', '.join(members[:-1]) + ', and ' + members[-1]
                    else:
                        formatted_members = ', '.join(members)
                    note_taker = [member['name'] for member in st.session_state.party_members if member.get('note_taker', False)][0]
                    response = self.llmhandler.invoke_model(self._PROMPTEMPLATE, {"question": user_question, "partymembers": formatted_members, "notes": notes, "notetaker": note_taker})  # Pass the query relevant note documents, party member names, and note taker name to the model

                    placeholder.empty()
                    references_found = True
                    with st.chat_message("assistant", avatar="🧙‍♂️"):

                    # Only display references if any were found
                        if(references_found):
                            response +="\n______________________________________________________\n"
                            response += "Note entry References: \n"
                            st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
                            st.write_stream(self.__stream_data(response))
                            # Create a unique button for each reference
                            for item in notes:
                                tempbuttoninfo.append([item.metadata["Date"],self.__reference_button, (item.page_content,), f"click_{st.session_state.button_key}"])
                                st.button(str(item.metadata["Date"]), on_click= self.__reference_button,args=(item.page_content,),  key = f"click_{st.session_state.button_key}")
                                time.sleep(0.02)
                                # Generate new button key for next button
                                st.session_state.button_key = st.session_state.button_key + 1

                            # Add reference button information for the response to the session state
                            st.session_state.buttoninfo.append(tempbuttoninfo)

                # Save canned response to chat history if no references
                else:
                    placeholder.empty()
                    response = "Could not find any relevant journal entries for your query. It could be that there is not any relevant information regarding your query in the notes, the question needs to be reworded, or spelling needs to be reviewed."
                    st.session_state.buttoninfo.append(None)
                    st.write_stream(self.__stream_data(response))
                    st.session_state.messages.append({"role": "assistant", "content": response, "avatar":"🧙‍♂️"})
                st.rerun()

    def __stream_data(self, response):
        for word in response.split(" "):
            yield word + " "
            time.sleep(0.02)

    # Define Streamlit dialog for reference content display
    @st.dialog("Reference Content")
    def __reference_button(self, content):
        st.write(content)

    def __delete_member(self, member_id):
        # Filter list to remove the specific ID
        st.session_state.party_members = [
            m for m in st.session_state.party_members if m['id'] != member_id
        ]

    def __toggle_note_taker(self, member_id):
        # Find the member and toggle their note_taker status
        if st.session_state['note_taker_' + member_id]:
            for member in st.session_state.party_members:
                if member['id'] == member_id:
                    member['note_taker'] = st.session_state['note_taker_' + member_id]
        else:
            for member in st.session_state.party_members:
                if member['id'] == member_id:
                    member['note_taker'] = False
                    break

    def __init_UI(self):
        st.title("TTRPG Journal Q&A Chatbot 🧙‍♂️")
        st.info("This app takes your notes from your TTRPG campaign and passes your question along with relevant context from your notes to the local LLM. It does not permanently store your notes or chat history or use them to train any model. Please consult provided references as the AI may hallucinate.")

    def run(self):
        self.__process_model_options()
        self.__process_journal_options()
        self.__update_message_history()
        self.__process_chat()
