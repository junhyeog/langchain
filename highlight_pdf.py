"""
pdf 파일에 하이라이트

problems:

"""

import base64
import datetime
import os
import re
import streamlit as st
from openai import OpenAI
import openai
import numpy as np
from scipy.spatial.distance import cosine
from apikey import apikey
from nltk.tokenize import sent_tokenize, word_tokenize
import colorsys
import uuid
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import fitz
from langchain_community.vectorstores import FAISS
from streamlit_javascript import st_javascript
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

openai.api_key = apikey
os.environ["OPENAI_API_KEY"] = apikey

client = OpenAI(api_key=apikey)
openAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
width = st_javascript("window.innerWidth")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def display_pdf(path):

    # Read file as bytes:
    # bytes_data = upl_file.getvalue()
    bytes_data = open(path, "rb").read()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8", "ignore")

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width={str(width)} height={str(width*4//3)}></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def display_queries():
    default_colors = ["#ffff00", "#ffc0cb", "#ffa500", "#00ff00", "#add8e6"]
    for query_info in st.session_state.queries:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            query_key = f"query_{query_info['uuid']}"
            st.text_input("Query:", key=query_key)
        with col2:
            color_key = f"color_{query_info['uuid']}"
            st.color_picker(
                "Color:", key=color_key, value=st.session_state.get(color_key, np.random.choice(default_colors))
            )
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
    st.session_state.queries = [query for query in st.session_state.queries if query["uuid"] != uuid]
    st.session_state.query_count -= 1


def blend_colors(highlights):
    # [('#ffff00', 0.5), ('#ff0000', 0.1), ('#00ff00', 0.3), ...] => (r, g, b, a)
    r = 0
    g = 0
    b = 0
    a = 0
    for color, alpha in highlights:
        r += int(color[1:3], 16)
        g += int(color[3:5], 16)
        b += int(color[5:7], 16)
        a += alpha

    a /= len(highlights)
    r /= len(highlights) * 255
    g /= len(highlights) * 255
    b /= len(highlights) * 255
    return r, g, b, a


def document_to_db(document: fitz.Document):
    sentences = []
    for page in document:
        text = page.get_text("text")
        _sentences = sent_tokenize(text)
        sentences.extend(_sentences)
        print(f"[+] sentences: {len(sentences)}")

    db = FAISS.from_texts(sentences, openAIEmbeddings, distance_strategy="MAX_INNER_PRODUCT", normalize_L2=True)
    print(f"[+] distance_strategy: {db.distance_strategy}, normalize_L2: {db._normalize_L2}")
    return db


def highlighte_pdf(document, to_be_updated, output_file=None):
    for chunk, color in to_be_updated.items():
        for page in document:
            rects = page.search_for(chunk)
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color[:3])
                annot.set_opacity(color[3])
                annot.update()

    if output_file is None:
        output_path = "./highlighted"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = os.path.basename(document.name)
        output_file = file_name.replace(".pdf", f"_{date}.pdf")
        output_file = os.path.join(output_path, output_file)
    document.save(output_file)

    display_pdf(output_file)

    # remove all annotations
    annot_found = 0
    for page in document:
        annot = page.first_annot
        while annot:
            annot_found += 1
            page.delete_annot(annot)
            try:
                annot = annot.next
            except:
                annot = None
    print(f"[+] {annot_found} annotations are removed.")
    return output_file


st.title("Semantic Search")

uploaded_file = st.file_uploader("Upload file:", type=["pdf"])
add_file = st.button("Add File")

similarity_threshold = st.slider("Similarity threshold", 0.1, 0.9, 0.2, 0.1)

if "query_count" not in st.session_state:
    add_query()

display_queries()
st.button("Add Query", on_click=add_query)

run_search = st.button("Search")

if add_file and not uploaded_file:
    st.warning("Please upload the file first.")
    exit()

if uploaded_file and add_file:
    bytes_data = uploaded_file.read()
    uploaded_path = "./uploaded"
    if not os.path.exists(uploaded_path):
        os.makedirs(uploaded_path)
    file_name = os.path.join(uploaded_path, uploaded_file.name)

    with open(file_name, "wb") as f:
        f.write(bytes_data)

    document = fitz.open(file_name)
    if "document" in st.session_state and st.session_state["document"] == document:
        st.warning("The file is already added.")
    else:
        st.session_state["document"] = document
        with st.spinner("Building a vecotr store..."):
            db = document_to_db(document)
            st.session_state["db"] = db
        st.success("File uploaded, chunked and embedded successfully")


if run_search:
    if "db" not in st.session_state:
        st.warning("Please add the file first.")
        exit()

    if st.session_state.get("queries", None) is None:
        st.warning("Please add the query first.")
        exit()

    query_info = []
    for _query_info in st.session_state.queries:
        query_key = f"query_{_query_info['uuid']}"
        color_key = f"color_{_query_info['uuid']}"
        query = st.session_state.get(query_key)
        color = st.session_state.get(color_key)
        if query:
            query_info.append((query, color))

    if len(query_info) == 0:
        st.warning("Please add the query first.")
        exit()

    with st.spinner("Searching..."):
        db = st.session_state["db"]
        document = st.session_state["document"]
        all_text = ""
        for page in document:
            all_text += page.get_text("text")
        to_be_updated = {}
        for query, color in query_info:
            # exact match using regex (case insensitive)
            matched = re.findall(re.escape(query), all_text, re.IGNORECASE)
            matched = list(set(matched))
            # print(f"[+] exact matched ({query}): {matched}")
            for match in matched:
                if to_be_updated.get(match) is None:
                    to_be_updated[match] = []
                to_be_updated[match].append((color, 1.0))

            # !  semantic match
            query_type = 2

            if query_type == 1:  # using pure query
                updated_query = query
            elif query_type == 2:  # using query and answer
                retriever = db.as_retriever(search_kwargs={"fetch_k": 8})
                retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
                updated_query = retrievalQA.run({"query": f"{query}"})
            elif query_type == 3:  # using prompt
                retriever = db.as_retriever(search_kwargs={"fetch_k": 8})
                retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
                prompt = f"""
                    I want to search for a sentence related to a query, or a sentence related to an answer to the query, among all the sentences in the document.
                    To perform a semantic search, it is assumed that the higher the cosine similarity with the embedding of the prompt, the more valid the search result is by comparing all sentences of the document with the embedding of an arbitrary prompt.
                    At this time, output the corresponding prompt as to which prompt to use for the most accurate search.
                    If you can't find a prompt, just output the query itself.
                    NOTICE: Just output the prompt only!
                    The query is as follows. query: {query}.
                """
                updated_query = retrievalQA.run({"query": f"{prompt}"})
            else:
                raise ValueError(f"Invalid query type: {query_type}")

            print(f"[+] update query: {query} -> {updated_query}")
            query = updated_query
            result = db.similarity_search_with_score(query, k=1000, score_threshold=similarity_threshold)
            # print(f"[+] semantic matched ({query}): {result}")
            for doc, similarity in result:
                chunk = doc.page_content
                if to_be_updated.get(chunk) is None:
                    to_be_updated[chunk] = []
                to_be_updated[chunk].append((color, similarity))

        # highlight
        for chunk, highlights in to_be_updated.items():
            color = blend_colors(highlights)
            to_be_updated[chunk] = color

    with st.spinner("Highlighting..."):
        output_file = highlighte_pdf(document, to_be_updated)
    st.success(f"Search completed: {output_file}")

else:
    st.warning("Document or query is missing.")
