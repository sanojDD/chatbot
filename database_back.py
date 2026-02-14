# -------------------------------
# Imports
# -------------------------------
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import sqlite3

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


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# # -------------------------------
# # Compile Workflow
# # -------------------------------
chatbot = graph.compile(checkpointer=checkpointer)

# test
# CONFIG = {"configurable": {"thread_id": "thread-1"}}
# response = chatbot.invoke({'messages': [HumanMessage(content="who am i")]},
#                           config=CONFIG)
# print(response)

#


def retrieve_all_threads():
  all_thread = set()
  for ckpt in checkpointer.list(None):
    all_thread.add((ckpt.config['configurable']['thread_id']))

  return list(all_thread)
