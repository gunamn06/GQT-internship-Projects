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

st.set_page_config(page_title="Multi-Document RAG QA")

st.title("Multi-Document Question Answering using RAG")

# ----------------------------
# Load Embeddings (cached)
# ----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ----------------------------
# Load LLM (cached)
# ----------------------------
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=512,
        temperature=0
    )
    return HuggingFacePipeline(pipeline=pipe)


uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

# If files uploaded
if uploaded_files and len(uploaded_files) > 0:

    try:
        all_documents = []

        with st.spinner(" Processing documents..."):

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                # Add filename metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name

                all_documents.extend(documents)
                os.remove(tmp_path)

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )

            docs = text_splitter.split_documents(all_documents)

            # Create embeddings
            embeddings = load_embeddings()

            # Create FAISS vector store
            vectorstore = FAISS.from_documents(docs, embeddings)

        st.success(f"{len(uploaded_files)} document(s) processed successfully!")

        query = st.text_input(" Ask a question from the uploaded documents:")

        if query:

            with st.spinner("Generating answer..."):

                llm = load_llm()

                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )

                result = qa_chain.invoke({"query": query})

                st.subheader(" Answer:")
                st.write(result["result"])

                st.subheader("Retrieved Chunks:")

                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.markdown(f"Source File: `{doc.metadata['source']}`")
                    st.write(doc.page_content)
                    st.write("---")

    except Exception as e:
        st.error(" Something went wrong:")
        st.error(str(e))

else:
    st.info("Please upload at least one PDF file to begin.")