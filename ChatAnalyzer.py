import os
import sys
import requests

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain_openai import ChatOpenAI

key = os.environ.get("OPENAI_API_KEY")

st.set_page_config("Blog Content Analyzer", page_icon=":memo:")

# Sidebar contents
with st.sidebar:
    st.title("Blog Analyzer 💬 ChatApp")

    st.markdown('''
    ## About
    This app is an OpenAI LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    
    st.write("A side project by [Jeevan Gupta](https://jeevangupta.com) ")


def extract_insights(webpage_content):
    try:
        insight = ""
        with open("./system_prompt.txt", "r") as f:
            system_prompt = f.read()

        main_prompt = """Here is the blog post: 
        <blog_post>{}</blog_post>"""

        messages = [
                {"role": "system", "content": system_prompt},
                {"role":"user", "content": main_prompt.format(webpage_content)}
            ]


        llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.5, max_tokens=1024)

        insight = llm.invoke(messages)
        insight = insight.content

        return insight
    except:
        error_message = f"\n Function extract_insights failed !!! {sys.exc_info()}"
        print(error_message)
        st.error(error_message)


def extract_webpage_content(url):
    try:
        text = ""
        # Make a GET request to the blog URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the content from the response
            text = response.text
        else:
            print('Failed to fetch content from the blog URL')

        return text
    except:
        error_message = f"\n Function extract_webpage_content failed !!! {sys.exc_info()}"
        print(error_message)
        st.error(error_message)
        


def main_setup():
    st.title("Chat with blog post content")

    # https://jeevangupta.com/python-tuples-tutorial-learning-python-tuples-with-example/
    url = st.text_input("Enter the blog URL:")
    
    # st.subheader("URL")
    # st.text_area("", value=url, height=100)

    blog_text = ""

    if url:
        blog_text = extract_webpage_content(url)
    else:
        st.warning("Please enter a valid URL.")
    
    # show url extracted content on UI
    st.subheader("Extracted text from URL")
    #st.text_area("", value=blog_text, height=300)
    with st.container(height=300, border= True):
        st.text(blog_text)


    if blog_text:
        start = st.button("Start Analysis", type="primary")

        if start:
            with st.spinner("Analyzing content..."):
                initial_insight = extract_insights(blog_text)
            
            with st.container(height=300, border= True):
                st.write(initial_insight)


if __name__ == "__main__":
    main_setup()