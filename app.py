import os
from apikey1 import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = apikey
st.title("Medium Article Generator")
topic = st.text_input("lnput your topic of interest")

title_template = PromptTemplate(
    input_variables=["topic", "language"],
    template="Give me medium article title on {topic} in {language}",
)

llm = OpenAI(temperature=0.9)
if topic:
    response = llm(title_template.format(topic=topic, language="english"))
    # response = llm(topic)
    st.write(response)
