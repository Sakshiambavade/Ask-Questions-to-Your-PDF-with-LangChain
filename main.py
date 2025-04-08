import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="🧠 Ask Your PDF", layout="wide")
st.title("📄 Ask Questions to Your PDF with LangChain")

# File upload
pdf = st.file_uploader("Upload a PDF", type="pdf")

# Choose embedding model
embedding_dim = st.selectbox("🔢 Select Embedding Dimension", [300, 700, 1500])

# Map dimension to model
model_map = {
    300: "sentence-transformers/all-MiniLM-L6-v2",
    700: "sentence-transformers/all-mpnet-base-v2",
    1500: "sentence-transformers/sentence-t5-large"
}
selected_model = model_map[embedding_dim]

if pdf is not None:
    # Extract text
    pdf_reader = PdfReader(pdf)
    raw_text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # Embeddings
    st.write(f"🔍 Generating embeddings with model: `{selected_model}`...")
    embeddings = HuggingFaceEmbeddings(model_name=selected_model)
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    st.success("✅ Vector store generated successfully!")

    # Question input
    question = st.text_input("🤖 Ask a question about the PDF:")
    if question:
        # LLM setup
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.2, "max_length": 512}
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        # Search documents
        docs = vectorstore.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)

        st.markdown("### 💬 Answer:")
        st.write(answer)
