#run: streamlit run streamlit_app.py

import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader

# Start gemini model and chat
genai.configure(api_key='AIzaSyBKQ_tbS34bR0Z6MM7j2iE5T4sEkZlM98k')
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat()

def send(message):
    try:
        response = chat.send_message(message).text
    except:
        st.warning("Query Blocked!", icon=':warning:')
        return "Query Blocked!"
    return response

def main():
    # Set the page title and layout
    st.set_page_config(page_title="Gemini ChatBot",
                    page_icon=":robot_face:",
                    layout="wide",
                    )
    st.title(':left_speech_bubble: :blue[AI ChatBot using Gemini] :left_speech_bubble:')

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
        with st.spinner('Loading the file...'):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
                else:
                    text += "Empty page"
            send(text)
        st.success('File Loaded!', icon=':white_check_mark:')
        st.balloons()
    
    if user_input:
        with st.spinner('Responding...'):
            # User chat
            with st.chat_message("user"):
                # Display input
                st.markdown(user_input)
                
                # Add to history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Create response
                response = f"ChatBot:\n {send(user_input)}"

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
