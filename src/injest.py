from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

#from langchain_openai import OpenAIEmbeddings (require API cost)
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# OpenAI API Key
# =====================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable must be set.")

# =====================================================
# Base Directory
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
print(f"base dir is : {BASE_DIR}")

# =====================================================
# Folder Paths
# =====================================================

resume_folder = BASE_DIR / "docs" / "resumes"
policy_folder = BASE_DIR / "docs" / "policies"
jd_folder = BASE_DIR / "docs" / "job_descriptions"
persist_directory = BASE_DIR / "chroma_db"


def load_documents_from_folder(folder: Path):
    documents = []
    if not folder.exists():
        print(f"Warning: missing folder, skipping: {folder}")
        return documents

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path))
        else:
            continue

        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source_folder"] = folder.name
            doc.metadata["file_name"] = path.name
        documents.extend(loaded_docs)

    return documents


def build_chroma_db():
    documents = []
    documents.extend(load_documents_from_folder(resume_folder))
    documents.extend(load_documents_from_folder(policy_folder))
    documents.extend(load_documents_from_folder(jd_folder))

    if not documents:
        raise FileNotFoundError(
            "No documents loaded. Check docs/resumes, docs/policies, or docs/job_descriptions for .txt/.pdf files."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    print(f"Total chunks created: {len(docs)}")

    embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=str(persist_directory),
    )
    vectordb.persist()
    print("ChromaDB created successfully")


if __name__ == "__main__":
    build_chroma_db()