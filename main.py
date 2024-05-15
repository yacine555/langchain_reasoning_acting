from typing import Union, List
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.tools.render import render_text_description

from langchain.chains import create_retrieval_chain

from backend.llm import get_llm, get_chatllm
from common.template import TEMPLATE_TOOLS

from backend.callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the lenght of the text in number of characters"""
    print(f"get_text_length enter with {text=}")
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")




if __name__ == "__main__":

    print("Hello Langchain RAG")

    tools = [get_text_length]

    # Quick start, create a basic chain with prompt, and parser
    llm = ChatOpenAI(
        temperature=0, model_kwargs={"stop":"\nObservation"}, callbacks=[AgentCallbackHandler()]
    )

    # Usecase 1: invoke llm with raw user input
    # llm.invoke("how can langsmith help with testing?")

    # Usecase 2: create a chain with a chat prompt template
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])

    output_parser = StrOutputParser()

    chain = prompt2 | llm | output_parser
    chain.invoke({"input": "how can langsmith help with testing?"})


    llm = get_llm(1,"gpt-4",0)
    
    # prompt = PromptTemplate.from_template(template=TEMPLATE_TOOLS).partial(
    #     tools=render_text_description(tools),
    #     tool_names=", ".join([t.name for t in tools]),
    # )


    # intermediate_steps = []
    # agent = (
    #     {
    #         "input": lambda x: x["input"],
    #         "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    #     }
    #     | prompt
    #     | llm
    #     | ReActSingleInputOutputParser()
    # )

    # agent_step = ""
    # while not isinstance(agent_step, AgentFinish):
    #     agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
    #         {
    #             "input": "What is the lenght in characters of DOG?",
    #             "agent_scratchpad": intermediate_steps,
    #         }
    #     )

    #     print(agent_step)

    #     if isinstance(agent_step, AgentAction):
    #         tool_name = agent_step.tool
    #         tool_to_use = find_tool_by_name(tools, tool_name)
    #         tool_input = agent_step.tool_input

    #         observation = tool_to_use.func(str(tool_input))
    #         print(f"{observation=}")
    #         intermediate_steps.append((agent_step, str(observation)))

    # if isinstance(agent_step, AgentFinish):
    #     print(agent_step.return_values)

    print("Finish")
