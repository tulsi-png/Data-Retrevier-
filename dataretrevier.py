import os
import streamlit as st
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔑 Set your Google Gemini API key here
os.environ["GOOGLE_API_KEY"] = "AIzaSyDtgwOZNPWmJ3cU-RV09cOnnAJAuj8JTxE"  # Replace with your actual key

# --- Streamlit UI ---
st.title("📄 Smart PDF Document Assistant")
st.markdown("Upload a PDF and ask specific questions like 'Extract project deadlines and team members'.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
user_query = st.text_input("User Query", placeholder="e.g., Extract project deadlines and team members")

if uploaded_file and user_query:
    # Step 1: Read PDF content
    pdf = PdfReader(uploaded_file)
    raw_text = ""
    pages = []

    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pages.append((i + 1, text))
            raw_text += text + "\n"

    # Step 2: Split & Embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([p[1] for p in pages], metadatas=[{"page": p[0]} for p in pages])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Step 3: RAG Prompt
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    template = """You are a smart document assistant.
A user has asked: "{question}"

Use only the relevant context from the PDF to answer.
If you don't find anything relevant, say: "No relevant information found in the PDF."

Only include content that is clearly supported by the document. Do not speculate.
Mention page numbers when possible.

Context:
{context}

Answer:"""

    rag_prompt = ChatPromptTemplate.from_template(template)

    rag_chain: Runnable = rag_prompt | model

    # Step 4: Query RAG
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n".join(
        [f"(Page {doc.metadata['page']}): {doc.page_content[:800]}..." for doc in retrieved_docs]
    )

    response = rag_chain.invoke({"question": user_query, "context": context})

    # Step 5: Display Result
    st.subheader("📌 Extracted Answer:")
    st.write(response.content)

    with st.expander("🔍 Source Excerpts"):
        for doc in retrieved_docs:
            st.markdown(f"**Page {doc.metadata['page']}**")
            st.write(doc.page_content[:1000])
