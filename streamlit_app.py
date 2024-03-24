from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate

# Google api key
GOOGLE_API_KEY = 'AIzaSyBKQ_tbS34bR0Z6MM7j2iE5T4sEkZlM98k'

# model for normal chatbot
genai.configure(api_key=GOOGLE_API_KEY)
model0 = genai.GenerativeModel(model_name="gemini-pro")

# model for documents chatbot
model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-pro")

def get_conversational_chain():

    prompt_template = """
    Be a helpful assistant and help with this pdf file and be open for any questions about it\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def get_response(user_question,file):
    if user_question.startswith("file:"):
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
        
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=200).split_text(text)

        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model = "models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain.run(input_documents=docs, question=user_question)
        return response
    else:
        response = model0.generate_content(user_question).text
        return response
    
def main():

    # Set the page title and layout
    st.set_page_config(page_title="Gemini ChatBot",
                    page_icon="🤖",
                    layout="wide",
                    )
    st.title(':rainbow[AI ChatBot using Gemini] :left_speech_bubble:')

    uploaded_file = st.file_uploader("Upload file ", type="pdf")
    user_input = st.chat_input("What is up?")
    st.write("Write 'file:' to access the file content")

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messagess
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input:
        with st.spinner('Responding...'):
            # User chat
            with st.chat_message("user"):
                # Display input
                st.markdown(user_input)
                
                # Add to history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Create response
                response = get_response(user_input,uploaded_file)
            
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
