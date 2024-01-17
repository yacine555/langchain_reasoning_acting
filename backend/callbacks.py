from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print("*****")
        print(f"***Prompt to LLM start***\n{prompts[0]}")
        print("*****")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print("*****")
        print(f"***LLM response***\n{response.generations[0][0].text}")
        print("*****")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
        ) -> Any:
        """Run when chain starts running."""
        print("*****CHAIN STARTS******")
        for key in serialized:
            print(f"   \n{key} : {serialized[key]}")
        print("*****")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        print("*****")
        print("*****CHAIN ENDS******")
        for key in outputs:
            print(f"   \n{key} : {outputs[key]}")
        print("*****")