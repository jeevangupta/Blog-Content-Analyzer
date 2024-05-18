import streamlit as st

def OnURLChange():
    st.session_state.web_url = st.session_state.url
    st.session_state.url = ""
    st.session_state.web_text = ""
    st.session_state.initial_insight = ""
    st.session_state.vector_store = {}
    st.session_state.start_analysis = False


def OnStartAnalysisCLick():
    st.session_state.start_analysis = True
    st.session_state.initial_insight = ""

def OnUserQueryChange():
    st.session_state.user_query = st.session_state.query
    st.session_state.query = ""