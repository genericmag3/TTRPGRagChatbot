import pandas as pd
from langchain_chroma import Chroma
import re
import io
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document as langchaindoc
from docx import Document as DocxReader

def load_embeddings():
    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": "cpu"}) # add check for cuda version and use 'cuda' if compatible?
    return embeddings

def parse_journal_text(file_content, databasedir):
    """Parses a text file with date headers into a structured list of dicts."""
    # Matches common date formats like 2023-10-27 or 10/27/2023 at the start of a line
    date_pattern = r'^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})'
    
    entries = []
    current_date = "Unknown Date"
    current_content = []

    for line in file_content.splitlines():
        match = re.match(date_pattern, line.strip())
        if match:
            # If we already have a previous entry, save it before starting a new one
            if current_content:
                entries.append({
                    "Title": f"Entry for {current_date}",
                    "Date": current_date,
                    "Contents": "\n".join(current_content).strip()
                })
            current_date = match.group(1)
            current_content = [line[match.end():].strip()] # Start content after the date
        else:
            current_content.append(line.strip())

    # Catch the final entry
    if current_content:
        entries.append({
            "Title": f"Entry for {current_date}",
            "Date": current_date,
            "Contents": "\n".join(current_content).strip()
        })
    return pd.DataFrame(entries)

def create_hf_retrival_artifacts(databasedir):
    hf_embeddings = load_embeddings()
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
    return text_splitter,retriever,vector_store


def convert_document_into_dataframe(document, databasedir):
    file_extension = document.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(document)
    elif file_extension == 'docx':
        # Read the file into a buffer
        bytes_data = document.read()
        doc_io = io.BytesIO(bytes_data)
        document = DocxReader(doc_io)

        # Create list of paragraphs
        document_text = []
        for paragraph in document.paragraphs:
            document_text.append(paragraph.text)
            
        # Join paragraphs together with newline character
        text_content = '\n'.join(document_text)

        # Parse text content into same dataframe structure
        df = parse_journal_text(text_content, databasedir)
    # simple text document
    else:
        # Read the text file content and parse it into the same dataframe structure
        stringio = io.StringIO(document.getvalue().decode("utf-8"))
        df = parse_journal_text(stringio.read(), databasedir)
    
    return df

# function yeilds progress percent until final returncode
def vectorize_note_document(document, databasedir, text_splitter, retriever, vector_store):
    df = convert_document_into_dataframe(document, databasedir)
    retCode = True

    if df is not None and not df.empty:
        documents = []
        idlist = []
        l = 0 
        
        for i, row in df.iterrows():
            text = str(row["Contents"])
            # Existing semantic chunking via text_splitter
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                document = langchaindoc(
                    page_content=chunk,
                    metadata={
                        "Title": row.get("Title", "Untitled"), 
                        "Date": str(row.get("Date", "Unknown")), 
                        "Exerpt Start": chunk[:25], 
                        "Exerpt End": chunk[-25:]
                    },
                    id=str(l)
                )
                idlist.append(str(l))
                documents.append(document)
                l += 1
            
            # Progress bar logic
            percent_complete = (i + 1) / len(df) * 100
            yield percent_complete # provide percent complete for progress bar usage
                
        if vector_store is not None:
            vector_store.add_documents(documents=documents, ids=idlist)
        else:
            retCode = False 
    else:
        retCode = False

    return retCode