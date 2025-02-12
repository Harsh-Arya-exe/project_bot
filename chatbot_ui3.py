import streamlit as st
import time
# from chatbot3 import ChatBot  # Ensure this filename matches your chatbot file name
from chatbot44 import ChatBot  # Ensure this filename matches your chatbot file name

# Initialize the chatbot once using Streamlit's cache.
@st.cache_resource
def get_chatbot():
    return ChatBot()

chatbot = get_chatbot()

# Configure the Streamlit page.
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("🤖 AI Research Assistant")
st.caption("Powered by Groq, LangGraph, and Streamlit")

# Custom CSS styling for the chat interface.
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 1rem; width: 95%}
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assistant-message {background-color: #f8f9fa; border-left: 4px solid #2ecc71;}
    .user-message {background-color: #e7f3fe; border-left: 4px solid #3498db;}
    .stSpinner > div {margin: auto;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for messages if not already set.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle chat input.
if prompt := st.chat_input("Ask me anything about AI research or general knowledge..."):
    # Append the user message to the session state.
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user message.
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get and display the assistant's response.
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            response = chatbot.run_chat(prompt)
            print("Final UI responce is : ",response.text)

            
            st.write("Answer: ", response.text)  # Ensure output is visible in UI
            response_placeholder.markdown(response.text)  # Update UI

                # summery.append(s)

            # if response.text == "":
            #     full_response = f"⚠️ Error in UI: {response.text}"
            #     print("if condition ")
            # else:
            #     print("else conditiion ", response.text)

            #     st.write("Final UI response is:", response.text)  # Ensure output is visible in UI
            #     response_placeholder.markdown(response.text)  # Update UI

                # Simulate streaming: update response word by word.
                # for chunk in response.split():
                #     full_response += chunk + " "
                #     time.sleep(0.05)
                #     response_placeholder.markdown(full_response + "▌")




        except Exception as e:
            full_response = f"⚠️ Critical error: {str(e)}"
        finally:
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
