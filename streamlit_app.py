from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
import streamlit as st
from PyPDF2 import PdfReader

# Google api key
GOOGLE_API_KEY = 'AIzaSyBKQ_tbS34bR0Z6MM7j2iE5T4sEkZlM98k'

# model for normal chatbot
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-pro")
chat = model.start_chat(history=[])

def load_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000,
                                            chunk_overlap=200).split_text(text)
    return chunks

def get_response(user_question):
    # use the file in the AI if the query starts with 'file:'
    if user_question.startswith("file:"):
        chunks = st.session_state.file
        
        # for small files use it as a whole
        context = '\n'.join(chunks)
        # for big files use embeddings and vector store to get chunks relevant to the question
        if len(chunks)>100:
            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model = "models/embedding-001")
            vector_store = FAISS.from_texts(chunks, embeddings)
            docs = vector_store.similarity_search(user_question)
            context = ""
            for doc in docs:
                context += doc.page_content + " "

        response = chat.send_message(f"""Answer the question based on the context: 
                                     Context: {context}
                                     Question: {user_question} 
                                     """).text
        
        return response
    # otherwise use the AI normally
    else:
        response = chat.send_message(user_question).text
        return response

# changes state
def isNew():
    st.session_state.is_new_file = True

def main():

    # Set the page title and layout
    st.set_page_config(page_title="Gemini ChatBot",
                    page_icon="🤖",
                    layout="wide",
                    )
    st.title(':rainbow[AI ChatBot using Gemini] :left_speech_bubble:')

    # Init boolean variable to check if there is a new file
    if "is_new_file" not in st.session_state:
        st.session_state.is_new_file = False

    if "file" not in st.session_state:
        st.session_state.file = None
    
    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    uploaded_file = st.file_uploader("Upload file ", type="pdf", on_change=isNew)
    user_input = st.chat_input("What is up?")
    st.write("Write \" file: \" to access the file content")

    # Display messagess
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Loading a new file
    if st.session_state.is_new_file and uploaded_file:
        with st.spinner("Loading File..."):
            
            st.session_state.file = load_pdf(uploaded_file) 
            st.session_state.is_new_file = False
            
            st.success("File Loaded!",icon="✅")
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
                try:
                    response = get_response(user_input)
                except:
                    response = "Error getting the response"
            
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
