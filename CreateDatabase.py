import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

df = pd.read_csv("Notes.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
#add_documents = not os.path.exists(db_location)

if True:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Contents"],
            metadata={"Title": row["Title"], "Date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="notes",
    persist_directory=db_location,
    embedding_function=embeddings
)

if True:
    vector_store.add_documents(documents=documents, ids=ids)