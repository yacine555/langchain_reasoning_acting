import os
from dotenv import load_dotenv

from langchain.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai.llms import OpenAI
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama

import sys
sys.path.append('/Volumes/Yacine T5/Evoke/Projects/Evoke/Tech/Tuorials/GenAI/GitHub/langchain_reasoning_acting/common')
import consts

# from common.consts import PENZZO_CONFIG

load_dotenv()

def get_llm(type:int, model_name:str, temperature=0, verbose=False,**kwargs) -> BaseLLM:
    """
        Type:   1 - OppenAI
                2 - Local Ollama
        model_name: name of the moodel to use

    """
    llm = None




    match type:
        case 1:
            llm = OpenAI(temperature=temperature, model=model_name, verbose = verbose)
        case 2:
            llm = Ollama(model=model_name,temperature = temperature,verbose=verbose)
        case _:
            print("The LLM type was not defined")

    
    return llm



def get_chatllm(type:int, model_name:str, temperature =0, verbose= False,**kwargs) -> BaseChatModel:
    """
        Type:   1 - OppenAI Chat 
                2 - Local ChatOllama
        model_name: name of the moodel to use

    """
    chatllm = None

    print(f"Chat env value API Key: {os.environ['OPENAI_API_KEY']}")
    
    has_penzzo_config = False
    if 'penzzo_log' in kwargs:
        has_penzzo_config = True
        
    match type:
        case 1:
            if has_penzzo_config:
                pezzo_config = consts.PENZZO_CONFIG["default_headers"]
                pezzo_config["X-Pezzo-Api-Key"] = os.environ['PEZZO_API_KEY']

                chatllm = ChatOpenAI(
                    temperature=temperature, 
                    model=model_name,
                    verbose=verbose,
                    openai_api_key = os.environ['OPENAI_API_KEY'],
                    openai_api_base = consts.PENZZO_CONFIG["openai_api_base"],
                    default_headers = pezzo_config
                )
            else:
                chatllm = ChatOpenAI(temperature=temperature, model=model_name, verbose = verbose)
        case 2:
            chatllm = ChatOllama(model=model_name,temperature = temperature,verbose=verbose)
        case _:
            print("The ChatLLM type was not defined")

    
    return chatllm


