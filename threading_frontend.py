import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid


# ----------------------------
# Utility Function
# ----------------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    return chatbot.get_state(config={
        "configurable": {
            "thread_id": thread_id
        }
    }).values['messages']


# ----------------------------
# Initialize Session Memory
# ----------------------------
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])

# ----------------------------
# Sidebar UI
# ----------------------------
st.header("Guffadi :blue[GPT] :sunglasses:")

st.sidebar.title('History', width="stretch", text_alignment="left")

if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('Chat History')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'

            temp_messages.append({"role": role, "content": message.content})

        st.session_state['message_history'] = temp_messages

# ----------------------------
# Display Previous Messages
# ----------------------------
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {"configurable": {"thread_id": st.session_state['thread_id']}}

# ----------------------------
# User Input
# ----------------------------

user_input = st.chat_input("Type here...")

if user_input:

    # Save user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.content for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ))

    # Save AI response
    st.session_state.message_history.append({
        "role": "assistant",
        "content": response
    })
