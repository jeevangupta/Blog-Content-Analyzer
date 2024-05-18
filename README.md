
# Blog Content Analyzer
    
- using LangChain and OpenAI API to extract insight and analyze a blog post given a URL of the blog page.

**Technical Underpinnings:**
- Text Chunking & OpenAI Embedding: Efficiently process text and identify relevant sections.
- Retrieval-Augmented Generation (RAG): Leverage advanced AI architectures for focused and accurate analysis.

## Getting Started Steps: 
1. create a virtual environment using the below command
    python3 -m venv env

2. activate virtual environment
    source env/bin/activate

3. create .env system file to store API key
    OPENAI_API_KEY="<replace with OpenAI API Key>"

4. use the below command to include the above .env file in the system
    source .env

5. Install all the needed library
    pip3 install -r requirements.txt

## Process without UI

- Run the blog-insights-extractor.py
    python3 ./blog-insights-extractor.py

## Process with UI

- Run ChatAnalyzer.py using the below command

    streamlit run ChatAnalyzer.py

**Here we use text chunking, and OpenAI embedding to search and return the top matching focused area based on the query, and then chain it to OpeinAI and Langchain QA to generate focused and accurate responses. This uses the Retrieval augmented generation, or RAG, which is an architectural approach.**
    
## Benefits for Content Consumers:
- Gain a deeper understanding of complex blog posts.
- Save time by quickly identifying key takeaways.
- Enhance research and information gathering processes.

## Future Enhancements:
- Sentiment Analysis: Explore adding sentiment analysis capabilities to gauge overall tone and message.
- Topic Modeling: Develop features to categorize and classify blog post themes.
- Comparative Analysis: Analyze and compare insights from multiple blog posts.

This project demonstrates the power of AI in content analysis, enabling users to extract valuable insights from online resources with greater efficiency.