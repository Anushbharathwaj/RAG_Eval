import RAG1.tools as t
from. tools import collection_info
from .prompts import system_agent_prompt
from .objects import Answer

from pprint import pprint

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature = 0)

#ReAct Agent Loop
agent = create_agent(
    model = llm,
    tools = [collection_info],
    system_prompt = system_agent_prompt,
    response_format = Answer
)

def ask(query : str) -> dict:
     """ Answer the user's question about the pdf"""

     response = agent.invoke({"messages":[{"role":"user",
                                "content": query}]})

     sr = response['structured_response']

     return {
        "answer" : sr.answer,
        "context" : sr.context
     }


#pprint(ask("Where do penguins live?"))