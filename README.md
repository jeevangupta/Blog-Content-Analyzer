
# Blog Content Analyzzer
    
- using LangChain and OpenAI API to extract insight and analyze a blog post given a URL of the blog page

## Getting Started Steps: 
1. create vertuial environment using below command
    python3 -m venv env

2. activate vertuial environment
    source env/bin/activate

3. create .env system file to store API key
    OPENAI_API_KEY="<replace with OpenAI API Key>"

4. use below command to include above .env file in system
    source .env

5. Install all the needed libary
    pip3 install -r requirements.txt

6. Run the blog-insights-extractor.py
    python3 ./blog-insights-extractor.py