import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Document QA using RAG", layout="wide")

st.title(" Document Question Answering System using RAG")

st.write("Upload one or more PDF files and ask questions based on the document content.")

# --------------------------------------------------
# Load Embeddings (Cached)
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

# --------------------------------------------------
# Load LLM (Cached)
# --------------------------------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)

# --------------------------------------------------
# File Upload 
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    try:
        all_documents = []

        with st.spinner("Processing documents..."):

            for uploaded_file in uploaded_files:

                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)

                os.remove(tmp_path)

            # --------------------------------------------------
            # Text Splitting
            # --------------------------------------------------
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,      # As per project explanation
                chunk_overlap=100    # To maintain context
            )

            docs = text_splitter.split_documents(all_documents)

            # --------------------------------------------------
            # Embeddings + FAISS
            # --------------------------------------------------
            embeddings = load_embeddings()

            vectorstore = FAISS.from_documents(
                docs,
                embeddings
            )

        st.success("Documents processed successfully!")

        # --------------------------------------------------
        # Question Input
        # --------------------------------------------------
        query = st.text_input("Enter your question:")

        if query:

            with st.spinner("Generating answer..."):

                llm = load_llm()

                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # Top-3 retrieval (Required)
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )

                result = qa_chain.invoke({"query": query})

                # --------------------------------------------------
                # Display Answer
                # --------------------------------------------------
                st.subheader(" Final Answer:")
                st.write(result["result"])

                # --------------------------------------------------
                # Display Source Chunks
                # --------------------------------------------------
                st.subheader("Retrieved Source Chunks:")

                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.markdown(f"Source File: `{doc.metadata['source']}`")
                    st.write(doc.page_content)
                    st.write("---")

    except Exception as e:
        st.error("Something went wrong.")
        st.error(str(e))

else:
    st.info("Please upload at least one PDF file to begin.")
