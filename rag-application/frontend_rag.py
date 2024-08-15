import streamlit as st
import rag_backend as backend

st.set_page_config(page_title="AWS EC2 FAQ using RAG")
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">AWS EC2 FAQ using RAG</p>'
st.markdown(new_title, unsafe_allow_html=True)

if 'vector_index' not in st.session_state: 
    with st.spinner("📀 Wait for the RAG magic..."):
        st.session_state.vector_index = backend.create_policy_index()

input_text = st.text_area("Enter your question", label_visibility="collapsed") 
go_button = st.button("🔍 Get Answer", type="primary")

if go_button: 
    with st.spinner("🔄 Processing your request... Please wait a moment."): 
        response_content = backend.retrieve_policy_response(index=st.session_state.vector_index, query=input_text)
        st.write(response_content)
