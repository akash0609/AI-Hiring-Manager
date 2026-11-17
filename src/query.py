import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_openai import ChatOpenAI


# =====================================================
# Load Environment Variables
# =====================================================

load_dotenv()


# =====================================================
# OpenAI API Key
# =====================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable must be set."
    )


# =====================================================
# Base Directory
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

print(f"Base dir is : {BASE_DIR}")


# =====================================================
# ChromaDB Path
# =====================================================

persist_directory = BASE_DIR / "chroma_db"


# =====================================================
# HuggingFace Embeddings
# =====================================================

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")


# =====================================================
# Load ChromaDB
# =====================================================

vectordb = Chroma(
    persist_directory=str(persist_directory),
    embedding_function=embedding
)

print("ChromaDB loaded successfully")


# =====================================================
# OpenAI LLM
# =====================================================

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

print("LLM loaded successfully")


# =====================================================
# Main RAG Function
# =====================================================

def run_hiring_analysis(query):


    # =====================================================
    # Retrieve Job Descriptions
    # =====================================================

    jd_docs = vectordb.similarity_search(
        query,
        k=2,
        filter={"source_folder": "job_descriptions"}
    )


    # =====================================================
    # Retrieve Policies
    # =====================================================

    policy_docs = vectordb.similarity_search(
        query,
        k=2,
        filter={"source_folder": "policies"}
    )


    # =====================================================
    # Retrieve Resumes With Scores
    # =====================================================

    resume_results = vectordb.similarity_search_with_score(
        query,
        k=8,
        filter={"source_folder": "resumes"}
    )


    # =====================================================
    # Similarity Threshold
    # Lower score = better match in Chroma
    # =====================================================

    SIMILARITY_THRESHOLD = 1.2


    # =====================================================
    # Filter Good Resume Matches
    # =====================================================

    resume_docs = []

    print("\n========== RESUME SCORES ==========\n")

    for doc, score in resume_results:

        print(f"\nScore: {score}")

        print(f"File Name: {doc.metadata.get('file_name')}")

        # Lower score = better match
        if score < SIMILARITY_THRESHOLD:

            resume_docs.append(doc)


    # =====================================================
    # No Suitable Match Found
    # =====================================================

    if len(resume_docs) == 0:

        return """
        No suitable candidates found for this role.

        The uploaded resumes do not appear to match
        the requested domain or skill set.
        """


    # =====================================================
    # Combine All Documents
    # =====================================================

    all_docs = jd_docs + policy_docs + resume_docs


    # =====================================================
    # Build Context
    # =====================================================

    context = ""

    print("\n========== RETRIEVED DOCUMENTS ==========\n")

    for i, doc in enumerate(all_docs):

        print(f"\n--- Document {i+1} ---\n")

        print(f"Source Folder: {doc.metadata.get('source_folder')}")

        print(f"File Name: {doc.metadata.get('file_name')}")

        print(doc.page_content[:700])

        context += f"""
        Source File: {doc.metadata.get('file_name')}

        Content:
        {doc.page_content}

        """


    # =====================================================
    # Final Prompt
    # =====================================================

    final_prompt = f"""
    You are an AI hiring assistant.

    Below are:
    - job description snippets
    - hiring policy snippets
    - candidate resume snippets

    Use ONLY the provided context.

    Evaluate candidates carefully.

    If no strong candidates exist,
    clearly mention that.

    Context:
    {context}

    Question:
    {query}

    Provide:
    1. Top matching candidates
    2. Why they are suitable
    3. Strengths
    4. Gaps
    5. Overall fit score out of 10
    """


    # =====================================================
    # Execute LLM
    # =====================================================

    print("\nRunning Final LLM Analysis...\n")

    response = llm.invoke(final_prompt)


    # =====================================================
    # Return Final Response
    # =====================================================

    return response.content