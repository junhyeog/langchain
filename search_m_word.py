"""
임베딩 단위를 문장 + 단어로 구성
"""
import streamlit as st
from openai import OpenAI
import openai
import numpy as np
from scipy.spatial.distance import cosine
from apikey1 import apikey
from nltk.tokenize import sent_tokenize, word_tokenize
import colorsys
import uuid

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

def update_word_embeddings(document):
    words = word_tokenize(document)
    word_embeddings = [get_embedding(word) for word in words]
    return words, word_embeddings


# default_colors = ["#ffff00", "#ffc0cb", "#ffa500", "#00ff00", "#add8e6"]
def display_queries():
    for query_info in st.session_state.queries:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            query_key = f"query_{query_info['uuid']}"
            st.text_input("Query:", key=query_key)
        with col2:
            color_key = f"color_{query_info['uuid']}"
            st.color_picker("Color:", key=color_key, value="#ffff00")
        with col3:
            remove_key = f"remove_{query_info['uuid']}"
            st.button(
                "Remove",
                key=remove_key,
                on_click=remove_query,
                args=(query_info["uuid"],),
            )


def add_query():
    if "query_count" not in st.session_state:
        st.session_state.query_count = 1
    else:
        st.session_state.query_count += 1
    query_info = {"uuid": str(uuid.uuid4())[:8]}
    if "queries" not in st.session_state:
        st.session_state.queries = []
    st.session_state.queries.append(query_info)


def remove_query(uuid):
    st.session_state.queries = [
        query for query in st.session_state.queries if query["uuid"] != uuid
    ]
    st.session_state.query_count -= 1

def blend_colors(highlights):
    # [('#ffff00', 0.5), ('#ff0000', 0.1), ('#00ff00', 0.3), ...] => (r, g, b, a)
    r = 0
    g = 0
    b = 0
    a = 0
    for color, alpha in highlights:
        r += int(color[1:3], 16) * alpha
        g += int(color[3:5], 16) * alpha
        b += int(color[5:7], 16) * alpha
        a += alpha
        
    a /= len(highlights)
    r = int(r / a)
    g = int(g / a)
    b = int(b / a)
    a = min(1, a)
    return r, g, b, a
    
st.title("Semantic Search")

document = st.text_area("Text:", height=300, key="document")
if "pre_document" not in st.session_state:
    st.session_state["pre_document"] = ""
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.2, 0.1)

if "query_count" not in st.session_state:
    add_query()
 
display_queries()
st.button("Add Query", on_click=add_query)


def update_document():
    # sentence
    sentences, sentence_embeddings = update_sentence_embeddings(document)
    st.session_state["pre_document"] = document
    st.session_state["sentences"] = sentences
    st.session_state["sentence_embeddings"] = sentence_embeddings
    # word
    words, word_embeddings = update_word_embeddings(document)
    st.session_state["words"] = words
    st.session_state["word_embeddings"] = word_embeddings


def check_and_update_document():
    if document != st.session_state["pre_document"]:
        update_document()


if document:
    check_and_update_document()
    highlighted_document = document
    for query_info in st.session_state.queries:
        query_key = f"query_{query_info["uuid"]}"
        color_key = f"color_{query_info["uuid"]}"
        query = st.session_state.get(query_key)
        to_be_updated = {}
        if query:
            query_embedding = get_embedding(query)
            
            # sentence
            chunks = st.session_state.get("sentences", [])
            chunk_embeddings = st.session_state.get("sentence_embeddings", [])
            # word
            chunks.extend(st.session_state.get("words", []))
            chunk_embeddings.extend(st.session_state.get("word_embeddings", []))
            
            for j, (chunk, chunk_embedding) in enumerate(zip(chunks,chunk_embeddings)):
                similarity = calculate_cosine_similarity(
                    query_embedding, chunk_embedding
                )
                if similarity > similarity_threshold:
                    
                    similarity = (similarity - similarity_threshold) / (
                        1 - similarity_threshold
                    )
                    if to_be_updated.get(chunk) is None:
                        to_be_updated[chunk] = []
                    color = st.session_state.get(color_key)
                    to_be_updated[chunk].append((color, similarity))

        for chunk, highlights in to_be_updated.items():
            color = blend_colors(highlights)
            highlighted_sentence = f'<span style="background-color: rgba({color[0]}, {color[1]}, {color[2]}, {color[3]});">{chunk}</span>'
            highlighted_document = highlighted_document.replace(chunk, highlighted_sentence)
    highlighted_document = highlighted_document.replace("\n", "<br>")
    st.markdown(highlighted_document, unsafe_allow_html=True)
else:
    st.warning("Input is missing. Please enter the text.")
