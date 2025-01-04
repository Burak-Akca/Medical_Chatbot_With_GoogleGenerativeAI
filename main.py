import streamlit as st
from langchain_helper import create_vector_database,get_chain
st.title("Burak Akça")
btn=st.button("Create Database")
if btn:
    create_vector_database()


question=st.text_input("Question: ")

if question:
    chain=get_chain()
    response=chain(question)
    st.header("Answer: ")
    st.write(response["result"])


