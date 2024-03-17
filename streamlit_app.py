import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader

def main():
    # Set the page title and layout
    st.set_page_config(page_title="Gemini ChatBot",
                    page_icon="🤖",
                    layout="wide",
                    )
    st.title(':left_speech_bubble: :blue[AI ChatBot using Gemini] :left_speech_bubble:')
    
    # Start gemini Chat
    genai.configure(api_key='AIzaSyBKQ_tbS34bR0Z6MM7j2iE5T4sEkZlM98k')
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat()

    uploaded_file = st.file_uploader("Upload file ", type="pdf")
    user_input = st.chat_input("What is up?")
    
    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messagess
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message ["content"])

    # Read the PDF and send it to the AI
    if uploaded_file:
        with st.spinner('Loading...'):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            chat.send_message(text)
            st.success('Done!')
            st.balloons()
    
    if user_input:
        # User chat
        with st.chat_message("user"):
            # Display input
            st.markdown(user_input)
            
            # Add to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Create response
            response = f"ChatBot:\n {chat.send_message(user_input).text}"

        # ChatBot chat
        with st.chat_message("assistant"):
            # Display response
            st.markdown(response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": response})

try:
    main()
except Exception as e:
    print(f"Error! {e}")
