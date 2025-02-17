import os
import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Load the API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    st.error("API key not found. Please set it in the environment variables.")
    st.stop()

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize ChatOpenAI and ConversationChain
llm = ChatOpenAI(model="gpt-4",  # Make sure to use a valid OpenAI model name here
                  openai_api_key=openai_api_key)

conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# Create user interface
st.title("🗣️ Conversational Chatbot")
st.subheader("㈻ Simple Chat Interface for LLMs by Build Fast with AI")

# User input and appending to the chat history
if prompt := st.chat_input("Your question"):  
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:  
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response if last message is from user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Generate the response from the LLM using the conversation chain
            response = conversation.predict(input=prompt)
            st.write(response)
            
            # Append assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
