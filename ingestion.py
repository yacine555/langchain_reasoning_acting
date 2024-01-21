import os
from typing import Any, Dict, List

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import DocusaurusLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings


from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import milvus
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

from common.consts import INDEX_NAME
from backend.llm import get_llm, get_chatllm

from backend.callbacks import AgentCallbackHandler
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain_community.callbacks import wandb_tracing_enabled


from loguru import logger

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "testwandb"


logfile = "./logs/output.log"
logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)
callback_manager = BaseCallbackManager([handler])

import pinecone
import chromadb


def ingest_docs(ingest_setup:int, index_name:str) -> Any:
    """
    ingest_setup: type of setup to implement React: the embeding, llm and vector database
        1: OpenAI with Pinecone with text
        2: Ollama llama2 with Chroma with PDF
        3: Ollama Phi with Faiss with site readTheDocs

    index_name: name of the index for the vectore store 
    """

    qa_vectorstore = None

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

                qa_vectorstore = Pinecone.from_documents(
                    texts, embeddings, index_name=index_name
                )

            else:
                print("Pinecone db already loaded DB loaded!")
                # index = pinecone.Index(index_name)
                # qa_vectorstore = Pinecone(index, embeddings, "text")

                qa_vectorstore = Pinecone.from_existing_index(
                    index_name=INDEX_NAME, embedding=embeddings
                )


            llm = get_llm(1,"gpt-4",0)
            qa_pinecone = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=qa_vectorstore.as_retriever()
            )

            query = "What is a vector DB? Give me a 15 word answer for a beginner"
            result = qa_pinecone.invoke({"query": query})

            print(result)

        case 2:

            embeddings_local = OllamaEmbeddings()

            persist_directory = "./vectorestore/chroma_db"

            if not os.path.exists(persist_directory):

                print("Loading documents into Chroma DB...")

                pdf_path = "./docs/2210.03629.pdf"
            
                loader_pdf= PyPDFLoader(file_path=pdf_path)
                documents_pdf = loader_pdf.load()

                print(f"loaded {len(documents_pdf)} documents")

                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
                pdf_doc = text_splitter.split_documents(documents_pdf)

                print(f"PDF Splitted into {len(pdf_doc)} chunks")

                qa_vectorstore = Chroma.from_documents(
                    documents=pdf_doc,
                    embedding=embeddings_local,
                    persist_directory=persist_directory,
                )
                qa_vectorstore.persist()
            else:
                qa_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings_local)
                print("Chroma DB loaded!")

            llm_locale = get_llm(2,"llama2",0)


            qa_chrome = RetrievalQA.from_chain_type(
                llm=llm_locale, chain_type="stuff", retriever=qa_vectorstore.as_retriever()
            )

            query = "What is a vector DB? Give me a 15 word answer for a beginner"
            result = qa_chrome.invoke({"query": query})
            print(result)

            # To cleanup, you can delete the collection
            qa_vectorstore.delete_collection()
            qa_vectorstore.persist()
            # Or just nuke the persist directory: !rm -rf db/

        case 3:
            print("Setting up FAISS VectoreStore!")

            embeddings_local = OllamaEmbeddings()
            faiss_vs = None

            faiss_path= "./vectorestore/faiss"

            if not os.path.exists(faiss_path):

                print("Loading documents into FAISS DB...")

                readthedoc_path2 = "./docs/python.langchain.com"
                readthedoc_path = "./docs/langchain.readthedocs.io2short"
                docusaurus_path = "https://python.langchain.com"

                # laod ReadTheDocs
                # loader_read_ted_docs = ReadTheDocsLoader(path=readthedoc_path)
                # documents_raw = loader_read_ted_docs.load()

                #load Docusaurus
                loader_docusaurus = DocusaurusLoader(docusaurus_path,verify_ssl=False)
                documents_raw = loader_docusaurus.load()

                print(f"loaded {len(documents_raw)} documents")

                #text_splitter_rec = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
                #raw_chunks = text_splitter_rec.split_documents(documents=documents_raw)
                raw_chunks = text_splitter.split_documents(documents=documents_raw)

                for chunk in raw_chunks:
                    old_path = chunk.metadata["source"]
                    new_url = old_path.replace("docs/","https://")
                    chunk.metadata.update({"source": new_url})


                print(f"Splitted into {len(raw_chunks)} chunks")


                # text_path = "./docs/vectorestore.txt"
            
                # loader = TextLoader(text_path)
                # documents = loader.load()

                # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
                # texts = text_splitter.split_documents(documents)

                qa_vectorstore = FAISS.from_documents(
                    documents=raw_chunks,
                    embedding=embeddings_local
                )

                qa_vectorstore.save_local(faiss_path)

            else:
                qa_vectorstore = FAISS.load_local(folder_path=faiss_path, embeddings = embeddings_local, index_name= "index")
                print("FAISS DB loaded!")

            llm_locale = get_llm(2,"phi",0)

            query="Give me the gist of React in 3 sentences"
            
            qa_faiss = RetrievalQA.from_chain_type(llm=llm_locale, chain_type="stuff", retriever=qa_vectorstore.as_retriever())
            result = qa_faiss.invoke({"query": query})
            print(result)

        case _:
            print("Default")


    return qa_vectorstore


def run_chatllm(rag_setup:int, query:str, chat_history: List[Dict[str,Any]]=[])  -> Any:

    qa_chain = None

    match rag_setup:
        case 1:
            print("Rag Pinecone!")

            embeddings = OpenAIEmbeddings()

            qa_vectorstore = Pinecone.from_existing_index(
                index_name=INDEX_NAME, embedding=embeddings
            )

            chat = get_chatllm(1,"gpt-3.5-turbo",0,True, penzzo_log=True)

            # qa_chain = RetrievalQA.from_chain_type(
            #     llm=chat,
            #     chain_type="stuff",
            #     retriever=qa_vectorstore.as_retriever(),
            #     return_source_documents=True,
            # )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=chat,
                chain_type="stuff",
                retriever=qa_vectorstore.as_retriever(),
                return_source_documents=True,
                callbacks=[handler],
                verbose=True
            )

        case 2:
            embeddings_local = OllamaEmbeddings()

            persist_directory = "./vectorestore/chroma_db"
            qa_vectorstore= Chroma(persist_directory=persist_directory, embedding_function=embeddings_local)

            chatllm_locale = chat = get_chatllm(2,"llama2",0,True, penzzo_log=True)

            # qa_chain = RetrievalQA.from_chain_type(
            #     llm=chatllm_locale, chain_type="stuff", retriever=qa_vectorstore.as_retriever(), return_source_documents=True
            # )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=chatllm_locale, chain_type="stuff", retriever=qa_vectorstore.as_retriever(), return_source_documents=True
            )


        case 3:
            embeddings_local = OllamaEmbeddings()

            qa_vectorstore = FAISS.load_local(folder_path="./faiss_index_react", embeddings = embeddings_local, index_name= "index")
            print("FAISS DB loaded!")

            chatllm_locale = get_chatllm(2,"phi",0,True)

            query="Give me the gist of React in 3 sentences"
            
            qa_chain = ConversationalRetrievalChain.from_llm(llm=chatllm_locale, chain_type="stuff", retriever=qa_vectorstore.as_retriever(), return_source_documents=True)

        case _:
            print("Default")


    print("=================")
    print(chat_history)
    print("=================")
    return qa_chain.invoke({"question": query, "chat_history": chat_history })


if __name__ == "__main__":

    print("Hello VectoreStore!")

    online_react_pinecone = False
    offline_react_chroma = False
    offline_react_faiss = False

    query_test = "What is a vector DB? Give me a 15 word answer for a beginner"
    qa_chain = run_chatllm(1, query_test)

    if online_react_pinecone:
        ingest_docs(1,"langchain-index")

    if offline_react_chroma:
       ingest_docs(2,"")

    if offline_react_faiss:
        ingest_docs(3,"faiss_index_react")
       


    

    
