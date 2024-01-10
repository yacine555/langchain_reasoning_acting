import os

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import pinecone
import chromadb

persist_directory = "db"
pinecone.init()

if __name__ == "__main__":
    print("Hello VectoreStore!")

    text_path = "./docs/vectorestore.txt"
    pdf_path = "./docs/2210.03629.pdf"
    
    loader = TextLoader(text_path)
    documents = loader.load()

    loader_pdf= PyPDFLoader(file_path=pdf_path)
    documents_pdf = loader_pdf.load()


    # print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    texts = text_splitter.split_documents(documents)
    pdf_doc = text_splitter.split_documents(documents_pdf)

    #print(len(texts))

    online_react_pinecone = False
    offline_react_chroma = False
    offline_react_faiss = True

    query = "What is a vector DB? Give me a 15 word answer for a beginner"

    if online_react_pinecone:
        embeddings = OpenAIEmbeddings()
        docsearch_vs = Pinecone.from_documents(
            texts, embeddings, index_name="langchain-index"
        )

        qa_pinecone = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=docsearch_vs.as_retriever()
        )

        result = qa_pinecone.invoke({"query": query})
        print(result)

    if offline_react_chroma:
        embeddings_local = OllamaEmbeddings()

        chrdb_vs = chromadb.from_documents(
            documents=texts,
            embedding=embeddings_local,
            persist_directory=persist_directory,
        )
        chrdb_vs.persist()

        llm_locale = Ollama(model="llama2")

        qa_chrome = RetrievalQA.from_chain_type(
            llm=llm_locale, chain_type="stuff", retriever=chrdb_vs.as_retriever()
        )
        result = qa_chrome.invoke({"query": query})
        print(result)

        # To cleanup, you can delete the collection
        chrdb_vs.delete_collection()
        chrdb_vs.persist()
        # Or just nuke the persist directory: !rm -rf db/

    if offline_react_faiss:

        is_document_loaded = True
        faiss_vs = None
        embeddings_local = OllamaEmbeddings()
        faiss_index_name = "faiss_index_react"

        if not is_document_loaded:

            faiss_vs = FAISS.from_documents(
                documents=texts,
                embedding=embeddings_local
            )

            faiss_vs.save_local(faiss_index_name)
        else:
            faiss_vs = FAISS.load_local(folder_path="./faiss_index_react", embeddings = embeddings_local, index_name= "index")

        llm_locale = Ollama(model="phi")

        query="Give me the gist of React in 3 sentences"
        
        qa_faiss = RetrievalQA.from_chain_type(llm=llm_locale, chain_type="stuff", retriever=faiss_vs.as_retriever())
        result = qa_faiss.invoke({"query": query})
        print(result)


    

    
