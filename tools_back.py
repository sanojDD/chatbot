# -------------------------------
# Imports
# -------------------------------
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()

# -------------------------------
# Initialize LLM
# -------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
  raise ValueError("GROQ_API_KEY not found in .env file")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# llm = ChatOpenAI(
#     model="stepfun/step-3.5-flash:free",
#     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
# )

google_llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview',
                                    temperature=0.7,
                                    google_api_key=GOOGLE_API_KEY)

# # -------------------------------
# # Tools
# # -------------------------------

search_tools = DuckDuckGoSearchRun(region='us-en')


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
  """
  Perform a basic arithmetic operation on two numbers.
  Supported operations: add, sub, mul, div
  """

  try:
    if operation == "add":
      result = first_num + second_num
    elif operation == "sub":
      result = first_num - second_num
    elif operation == "mul":
      result = first_num * second_num
    elif operation == "div":
      if second_num == 0:
        return {"error": "Division by zero is not allowed"}
      result = first_num / second_num
    else:
      return {"error": f"Unsupported operation '{operation}'"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result
    }

  except Exception as e:
    return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
  """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
  url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=YC46ZCZR3NN217A1"
  r = requests.get(url)
  return r.json()


tools = [search_tools, calculator, get_stock_price]
llm_with_tools = google_llm.bind(tools=tools)

# # -------------------------------
# # State
# # -------------------------------


class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


# # -------------------------------
# # Nodes
# # -------------------------------


def chat_node(state: ChatState):
  """
  llm node that may may answer or request a tool call
  """
  messages = state['messages']
  response = llm_with_tools.invoke(messages)
  return {"messages": [response]}


# Checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

tool_node = ToolNode(tools)
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

# # -------------------------------
# # Compile Workflow
# # -------------------------------
chatbot = graph.compile(checkpointer=checkpointer)

# test
# CONFIG = {"configurable": {"thread_id": "thread-1"}}
# response = chatbot.invoke({'messages': [HumanMessage(content="who am i")]},
#                           config=CONFIG)
# print(response)

#Helper Functions


def retrieve_all_threads():
  all_thread = set()
  for ckpt in checkpointer.list(None):
    all_thread.add((ckpt.config['configurable']['thread_id']))

  return list(all_thread)
