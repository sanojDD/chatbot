# -------------------------------
# Imports
# -------------------------------
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

# -------------------------------
# Initialize LLM
# -------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
  raise ValueError("GROQ_API_KEY not found in .env file")

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
)


class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


# # -------------------------------
# # Nodes
# # -------------------------------


def chat_node(state: ChatState):
  messages = state['messages']
  response = llm.invoke(messages)
  return {"messages": [response]}


# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# # -------------------------------
# # Compile Workflow
# # -------------------------------
chatbot = graph.compile(checkpointer=checkpointer)
