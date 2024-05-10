import os
import sys
import requests

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


key = os.environ.get("OPENAI_API_KEY")

def extract_webpage_content(url):
    try:
        content = ""
        # Make a GET request to the blog URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Get the content from the response
            content = response.text
        else:
            print('Failed to fetch content from the blog URL')

        return content
    except:
        print(f"\n Function extract_webpage_content failed !!! {sys.exc_info()}")


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
        print(f"\n Function extract_insights failed !!! {sys.exc_info()}")



if __name__ == '__main__':

    blog_url = 'https://jeevangupta.com/python-dictionary-tutorial/'

    blog_content = extract_webpage_content(blog_url)
    #print(blog_content)
    with open("./data/webpage_content.txt", 'w', encoding='utf-8') as file:
        file.write(blog_content)

    
    blog_insight = extract_insights(blog_content)
    with open("./data/webpage_insight.txt", 'w', encoding='utf-8') as file:
        file.write(blog_insight)