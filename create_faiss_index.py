# pip install streamlit pymupdf langchain faiss-cpu sentence-transformers
import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

INDEX_FILE_PATH = "microbio_faiss_index"
PDF_PATH = "C:\\Users\\User\\Downloads\\Microbiology and Immunology Textbook of 2nd Edition ( PDFDrive ).pdf"

def create_faiss_index(docs, index_path):
    embeddings = HuggingFaceEmbeddings()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]

    if texts:
        print("範例段落：")
        print(texts[0][:300])

    vector_store = FAISS.from_texts(texts, embeddings)
    vector_store.save_local(index_path)
    print(f"儲存成功：{index_path}")
    return vector_store



if os.path.exists(INDEX_FILE_PATH):
    print("已存在向量庫，嘗試讀取...")
    vector_store = FAISS.load_local(INDEX_FILE_PATH, embeddings=HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    print("成功讀取 FAISS 向量庫")
else:
    print("嘗試讀取 PDF 檔案...")
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"PDF載入，共 {len(docs)} 頁")

    if docs:
        vector_store = create_faiss_index(docs, INDEX_FILE_PATH)
    else:
        vector_store = None

