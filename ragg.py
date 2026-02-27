import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="📄 Smart RAG QA System",
    page_icon=":)",
    layout="wide"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .answer-box {
            background-color: #f0f9f4;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Multi-Document RAG Question Answering</div>', unsafe_allow_html=True)
st.write("")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("⚙ Settings")
st.sidebar.write("Upload PDFs and ask questions.")

show_chunks = st.sidebar.checkbox("Show Retrieved Chunks")

# ----------------------------
# Cached Embeddings
# ----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------------
# Cached LLM
# ----------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=300,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)

# ----------------------------
# File Upload
# ----------------------------
uploaded_files = st.file_uploader(
    "📂 Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    try:
        all_documents = []

        with st.spinner("🔄 Processing documents..."):

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)
                os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )

            docs = text_splitter.split_documents(all_documents)

            embeddings = load_embeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

        st.success("Documents processed successfully!")

        st.divider()

        query = st.text_input("Ask your question:")

        if query:

            with st.spinner(" Generating answer..."):

                llm = load_llm()
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )

                result = qa_chain.invoke({"query": query})

                # Answer Section
                st.subheader(" Answer")
                st.markdown(f'<div class="answer-box">{result["result"]}</div>', unsafe_allow_html=True)

                # Retrieved Chunks Section
                if show_chunks:
                    st.subheader("Retrieved Chunks")

                    for i, doc in enumerate(result["source_documents"]):
                        with st.expander(f"Chunk {i+1} - {doc.metadata['source']}"):
                            st.write(doc.page_content)

    except Exception as e:
        st.error( "Something went wrong")
        st.error(str(e))

else:
    st.info("📂 Please upload at least one PDF file to begin.")