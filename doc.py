import os
from apikey1 import apikey
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = apikey


def clear_history():
    if "history" in st.session_state:
        del st.session_state["history"]


st.title("Chat with Document")  # title in our web page
uploaded_file = st.file_uploader("Upload file:", type=["pdf", "docx", "txt"])
add_file = st.button("Add File", on_click=clear_history)

if uploaded_file and add_file:
    bytes_data = uploaded_file.read()
    file_name = os.path.join("./", uploaded_file.name)

    with open(file_name, "wb") as f:
        f.write(bytes_data)

    loader = TextLoader(file_name)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    # initialize OpenAI instance
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = vector_store.as_retriever()

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    st.session_state.crc = crc

    # success message when file is chunked & embedded successfully
    st.success("File uploaded, chunked and embedded successfully")

# get question from user input
question = st.text_input("Input your question")

if question:
    if "crc" in st.session_state:
        crc = st.session_state.crc
        if "history" not in st.session_state:
            st.session_state["history"] = []

        response = crc.run(
            {"question": question, "chat_history": st.session_state["history"]}
        )

        st.session_state["history"].append((question, response))
        st.write(response)
        for prompts in st.session_state["history"]:
            st.write("Question:  \n", prompts[0])
            st.write("Answer:  \n", prompts[1])
