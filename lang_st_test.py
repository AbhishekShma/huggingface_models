from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import streamlit as st


api_token = st.secrets["API_TOKEN"]
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    huggingfacehub_api_token=api_token,
    max_new_tokens=100,
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Prompt template
template = PromptTemplate(
    template="{input}",
    input_variables=["input"]
)

st.set_page_config(page_title="Chat", layout="wide")
st.header("Chat")

# Floating input
st.markdown("""
    <style>
        .chat-input-container {
            position: fixed;
            bottom: 20px;
            left: 50px;
            right: 50px;
            background-color: #ffffff;
            padding: 12px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.15);
            z-index: 9999;
        }
        .chat-history {
            margin-bottom: 80px;
        }
    </style>
""", unsafe_allow_html=True)

# state for displaying chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# history
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Bot:** {message['content']}")
st.markdown('</div>', unsafe_allow_html=True)

# Process user input
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        prompt = template.invoke({"input": user_input})
        result = model.invoke(prompt)
        st.session_state.messages.append({"role": "bot", "content": result.content})
        st.session_state.user_input = ""  # Clear after sending

# Floating input at bottom
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
st.text_input("Type your message and press Enter", key="user_input", on_change=handle_input)
st.markdown('</div>', unsafe_allow_html=True)
