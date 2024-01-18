from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent, initialize_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.agents import create_csv_agent
from langchain.agents.types import AgentType

from langchain_openai.llms import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama


def python_agent() -> any:
    print("Start Agent")


    llm_local = Ollama(model="mistral",temperature=0,verbose=True)
    llm=ChatOpenAI(temperature=0, model="gpt-4")

    python_agent_executor = create_python_agent(
        llm=llm_local,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # python_agent_executor.invoke(
    #     """generate and save in current working directory 3 QRcodes
    #     that point to www.coursera.com, you have qrcode package installed already"""
    # )

    return python_agent_executor


def csv_agent() ->any:

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="./docs/episode_info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent.invoke("how many columns are there in file episode_info.csv")
    # csv_agent.invoke("print seasons ascending order of the number of episodes they have")

    return csv_agent

def agent_router():

    tools=[
        Tool(
            name="PythonAgent",
            func=python_agent().invoke,
            description="""useful when you need to transform natural language and write from it python and execute the python code,
                            returning the results of the code execution,
                        DO NOT SEND PYTHON CODE TO THIS TOOL""",
        ),
        Tool(
            name="CSVAgent",
            func=csv_agent().invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                            takes an input the entire question and returns the answer after running pandas calculations""",
        ),
    ]
    llm=ChatOpenAI(temperature=0, model="gpt-4",verbose=True)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    rout_agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=rout_agent, tools=tools, verbose=True)

    agent_executor.invoke({"input": "print seasons ascending order of the number of episodes they have"})

if __name__ == "__main__":
    #python_agent()
    #csv_agent()
    agent_router()