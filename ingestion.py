import os

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
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






def ingest_docs(ingest_setup:int, index_name:str)->None:
    """
    ingest_setup: type of setup to implement React: the embeding, llm and vector database
        1: OpenAI with Pinecone with text
        2: Ollama llama2 with Chroma with PDF
        3: Ollama Phi with Faiss with site readTheDocs

    index_name: name of the index for the vectore store 
    """


    match ingest_setup:
        case 1:
            print("Setting up Pinecone VectoreStore!")

            embeddings = OpenAIEmbeddings()
            pinecone_vs = None

            pinecone.init()
            index_list = pinecone.list_indexes()
            index_description = pinecone.describe_index(index_name)
            #index = pinecone.Index(index_name)
            
            pinecone_index = Pinecone.get_pinecone_index(index_name = index_name)
            index_description_stats = pinecone_index.describe_index_stats()

            if index_description_stats.total_vector_count == 0:
                print("Pinecone is empty! loading....")

                text_path = "./docs/vectorestore.txt"
            
                loader = TextLoader(text_path)
                documents = loader.load()

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
                texts = text_splitter.split_documents(documents)

                print(f"Text Splitted into {len(texts)} chunks")

                pinecone_vs = Pinecone.from_documents(
                    texts, embeddings, index_name=index_name
                )

            else:
                print("Pinecone db already loaded DB loaded!")
                index = pinecone.Index(index_name)
                embeddings = OpenAIEmbeddings()
                pinecone_vs = Pinecone(index, embeddings, "text")


            qa_pinecone = RetrievalQA.from_chain_type(
                llm=OpenAI(), chain_type="stuff", retriever=pinecone_vs.as_retriever()
            )

            query = "What is a vector DB? Give me a 15 word answer for a beginner"
            result = qa_pinecone.invoke({"query": query})

            print(result)

        case 2:
            print("Setting up Chroma VectoreStore!")


            embeddings_local = OllamaEmbeddings()

            persist_directory = "./chroma_db"
            chroma_vs= Chroma(persist_directory=persist_directory, embedding_function=embeddings_local)

            if chroma_vs == None:

                pdf_path = "./docs/2210.03629.pdf"
            
                loader_pdf= PyPDFLoader(file_path=pdf_path)
                documents_pdf = loader_pdf.load()

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
                pdf_doc = text_splitter.split_documents(documents_pdf)

                print(f"PDF Splitted into {len(pdf_doc)} chunks")

                chroma_vs = Chroma.from_documents(
                    documents=pdf_doc,
                    embedding=embeddings_local,
                    persist_directory=persist_directory,
                )
                chroma_vs.persist()
            else:
                print("Chroma DB loaded!")

            llm_locale = Ollama(model="llama2")

            qa_chrome = RetrievalQA.from_chain_type(
                llm=llm_locale, chain_type="stuff", retriever=chroma_vs.as_retriever()
            )

            query = "What is a vector DB? Give me a 15 word answer for a beginner"
            result = qa_chrome.invoke({"query": query})
            print(result)

            # To cleanup, you can delete the collection
            chroma_vs.delete_collection()
            chroma_vs.persist()
            # Or just nuke the persist directory: !rm -rf db/

        case 3:
            print("Setting up FAISS VectoreStore!")

            readthedoc_path = "./docs/python.langchain.com"

            loader_read_ted_docs = ReadTheDocsLoader(path=readthedoc_path)
            documents_raw = loader_read_ted_docs.load()

            print(f"loaded {len(documents_raw)} documents")

            text_splitter_rec = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
            raw_docs = text_splitter_rec.split_documents(documents=documents_raw)

            print(f"Splitted into {len(raw_docs)} chunks")


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

        case _:
            pass



if __name__ == "__main__":

    print("Hello VectoreStore!")

    online_react_pinecone = True
    offline_react_chroma = False
    offline_react_faiss = False

    if online_react_pinecone:
        ingest_docs(1,"langchain-index")

    if offline_react_chroma:
       ingest_docs(2,"")

    if offline_react_faiss:
        ingest_docs(3,"faiss_index_react")
       


    

    
