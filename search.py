import streamlit as st
from openai import OpenAI
import openai
import numpy as np
from apikey1 import apikey
from nltk.tokenize import sent_tokenize, word_tokenize

openai.api_key = apikey

client = OpenAI(api_key=apikey)


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return np.array(embedding)


def calculate_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def update_sentence_embeddings(document):
    sentences = sent_tokenize(document)
    sentence_embeddings = [get_embedding(sentence) for sentence in sentences]
    return sentences, sentence_embeddings


st.title("Semantic Search")

search_query = st.text_input("Query:", key="query")
document = st.text_area("Text:", height=300, key="document")
st.session_state["pre_document"] = ""
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.2, 0.1)


def update_document():
    sentences, sentence_embeddings = update_sentence_embeddings(document)
    st.session_state["pre_document"] = document
    st.session_state["sentences"] = sentences
    st.session_state["sentence_embeddings"] = sentence_embeddings


def check_and_update_document():
    if document != st.session_state["pre_document"]:
        update_document()


if search_query and document:
    query_embedding = get_embedding(search_query)
    check_and_update_document()
    highlighted_document = st.session_state["pre_document"]
    for sentence, sentence_embedding in zip(
        st.session_state.get("sentences", []),
        st.session_state.get("sentence_embeddings", []),
    ):
        similarity = calculate_cosine_similarity(query_embedding, sentence_embedding)
        print(similarity, sentence)
        if similarity > similarity_threshold:
            similarity = (similarity - similarity_threshold) / (
                1 - similarity_threshold
            )
            highlighted_sentence = f'<span style="background-color: rgba(255, 255, 0, {similarity});">{sentence}</span>'
            print("sentence", sentence)
            print("highlighted_sentence", highlighted_sentence)
            highlighted_document = highlighted_document.replace(
                sentence, highlighted_sentence
            )

    highlighted_document = highlighted_document.replace("\n", "<br>")

    st.markdown(highlighted_document, unsafe_allow_html=True)
else:
    st.warning("Input is missing. Please enter both query and text.")
