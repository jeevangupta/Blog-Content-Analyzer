import os
import sys
import requests
import pickle

import urllib.parse

from SessionStateFunction import *

import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_core.vectorstores import VectorStoreRetriever

from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

key = os.environ.get("OPENAI_API_KEY")

st.set_page_config("Blog Content Analyzer", page_icon=":speech_balloon:")

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
        raw_data = ""
        # Make a GET request to the blog URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the content from the response
            raw_data = response.text
            with open("./data/raw_data.txt","w") as f:
                f.write(raw_data)
        else:
            print('Failed to fetch content from the blog URL')

        return raw_data
    except:
        error_message = f"\n Function extract_webpage_content failed !!! {sys.exc_info()}"
        print(error_message)
        st.error(error_message)
        

def chunk_and_embed_data(blog_text, embedding_file_name):
    try:
        #print(embedding_file_name)
        loader = TextLoader("./data/raw_data.txt")
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function=len
            )
        #chunks = text_splitter.split_text(text=blog_text)
        chunks = text_splitter.split_documents(document)
        #st.write(chunks)
        
        embeddings = OpenAIEmbeddings()
        if os.path.exists(f"./data/{embedding_file_name}.pkl"):
            vector_store.load_local(embedding_file_name, embeddings)

        else:
            #vector_store = FAISS.from_texts(chunks, embeddings)
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            vector_store.save_local(embedding_file_name)

        return vector_store
    except:
        error_message = f"\n Function chunk_and_embed_data failed !!! {sys.exc_info()}"
        print(error_message)
        st.error(error_message)


def blog_analyzer_chat(query, vector):
    try:
        db = vector.similarity_search(query=query, k=3)
        #retriever = db.as_retriever()
        #st.write(docs)

        chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=db, question=query)
            #print(cb)
            with st.container(height=200, border= True):
                st.write(response)

    except:
        error_message = f"\n Function blog_analyzer_QA_chat failed !!! {sys.exc_info()}"
        print(error_message)
        st.error(error_message)


def main_setup():
    st.title("Chat with ULR (blog post) content")

    # https://jeevangupta.com/about-jeevan-gupta/
    # https://jeevangupta.com/python-tuples-tutorial-learning-python-tuples-with-example/
    # https://en.wikipedia.org/wiki/State_of_the_Union#:~:text=Though%20the%20language%20of%20the,as%20late%20as%20March%207
    
    if 'web_url' not in st.session_state:
        st.session_state['url'] = ''
        st.session_state['web_url'] = ''
        st.session_state['web_text'] = ''
        st.session_state['url_change'] = False
        st.session_state['initial_insight'] = ''
        st.session_state['vector_store'] = {}
        st.session_state['start_analysis'] = False
        st.session_state['query'] = ''
        st.session_state['user_query'] = ''
        

    url = st.text_input("Enter the blog URL:", key="url", on_change=OnURLChange)
    # st.text_area("", value=url, height=100)
    
    embedding_file_name = ""
    blog_text = ""
    if st.session_state.web_url:
        st.write(f" URL: {st.session_state.web_url}")

        ## extract domain name from url
        parsed_url = urllib.parse.urlparse(st.session_state.web_url)
        domain_name = parsed_url.netloc.split(":")[0]
        embedding_file_name = domain_name.replace(".com","")
        #st.write(domain_name)

        blog_text = extract_webpage_content(st.session_state.web_url)
        st.session_state['web_text'] = blog_text

        # show url extracted content on UI
        st.subheader("Extracted text from URL")
        if st.session_state.web_text:
            with st.container(height=300, border= True):
                st.text(st.session_state.web_text)

        #Extract general analysis based on inbuilt default prompt
        if not st.session_state.initial_insight:
            #st.write(st.session_state.start_analysis)
            start = st.button("Start Analysis", type="primary", on_click=OnStartAnalysisCLick)
            #st.write(st.session_state.start_analysis)

            if start or st.session_state.start_analysis:
                with st.spinner("Analyzing content..."):
                    initial_insight = extract_insights(blog_text)
                    st.session_state['initial_insight'] = initial_insight
        
        #display general insight on UI
        if st.session_state.initial_insight:
            # show url extracted content on UI
            st.subheader("General URL content insight")
            with st.container(height=300, border= True):
                st.write(st.session_state.initial_insight)

        #start Chat Analyzer
        if not st.session_state.user_query:
            start_chart = st.button("Start Chat", type="primary")
            if start_chart:
                with st.spinner("Chunking and Embedding content..."):
                    vector = chunk_and_embed_data(blog_text, embedding_file_name)
                    st.session_state['vector_store'] = vector

        if st.session_state.vector_store:
            st.subheader("Chat Analyzer")
            #accept user questions/query
            query = st.text_input("Ask questions about the URL (Blog post)", key="query", on_change=OnUserQueryChange)
            st.write(f"query: {st.session_state.user_query}")
            
            if st.session_state.user_query:
                with st.spinner("working on it..."):
                    blog_analyzer_chat(st.session_state.user_query, st.session_state.vector_store)
    else:
        st.warning("Please enter a valid URL.")
    

if __name__ == "__main__":
    main_setup()