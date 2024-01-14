from typing import Any
from common.consts import INDEX_NAME

from langchain_openai.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA

import pinecone
import chromadb


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOllama(verbose=True, temperature=0, model="llama2")
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    
    return qa({"query": query})