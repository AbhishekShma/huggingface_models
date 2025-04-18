from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
import streamlit as st

api_token = st.secrets["API_TOKEN"]
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    huggingfacehub_api_token=api_token,
    max_new_tokens=100,  
    task="text-generation"  
)


model = ChatHuggingFace(llm=llm)


st.header("Chat")
#user_input = st.text_input("Ask anything")


template = PromptTemplate(template=

"{input} Limit to 3 sentences."                                               
,
input_variables=["input"])



# start here
# user input state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Take input
user_input = st.text_input("Ask anything", value=st.session_state.get('user_input', ''), key="unique_key_1")

# Update state
if user_input:
    st.session_state.user_input = user_input

# Check if there's any input and the user pressed enter
if st.session_state.user_input:
    prompt = template.invoke({"input": st.session_state.user_input})
    result = model.invoke(prompt)
    st.write(result.content)

