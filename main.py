import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
from bs4 import BeautifulSoup 
from dotenv import load_dotenv
import os
import time


load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Function to extract article content
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get title
        title = soup.find('title').get_text() if soup.find('title') else "No title found"
        
        # Get paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return {"title": title, "content": text}
    except Exception as e:
        return {"error": f"Error extracting news from {url}: {e}"}

# Prompt templates
summarize_prompt = PromptTemplate(
    template="""Summarize the following news article in a concise way:
    Title: {title}
    Content: {content}
    
    Provide the summary in following format:
    Key Points:
    - Point 1
    - Point 2
    
    Brief Summary:
    [2-3 sentences summary]""",
    input_variables=["title", "content"]
)

# Chain to generate summary
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Streamlit UI
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
        .title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            color: #1f4e79;
            margin-bottom: 10px;
            padding: 20px;
            background: #f0f5f9;
            border-radius: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 22px;
            margin-bottom: 30px;
            color: #666;
        }
        .stButton button {
            background-color: #1f4e79;
            color: white;
            font-weight: bold;
            padding: 10px 25px;
            border-radius: 5px;
            width: 100%;
        }
        .gradient-text {
            background: linear-gradient(45deg, #1f4e79, #4CAF50);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            border-left: 5px solid #1f4e79;
        }
        .stats-box {
            background: linear-gradient(135deg, #f0f5f9, #e1e8f0);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            transition: transform 0.2s;
        }
        .stats-box:hover {
            transform: translateY(-5px);
        }
        .animate-pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">📰 Smart News Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Get quick, intelligent summaries of any news article</div>', unsafe_allow_html=True)

# Create two columns for the layout
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### Enter Article Details")
    user_url = st.text_input("News Article URL", placeholder="https://example.com/news-article")
    
    with st.expander("Advanced Options"):
        summary_length = st.select_slider(
            "Summary Length",
            options=["Short", "Medium", "Long"],
            value="Medium"
        )
    
    # Replace the existing spinner with this enhanced version
    if st.button("Generate Summary"):
        if user_url:
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            with st.spinner("🤖 AI is analyzing your article..."):
                # Extract article
                article_data = extract_news(user_url)
                
                if "error" in article_data:
                    st.error(article_data["error"])
                else:
                    with col2:
                        st.markdown("### Summary Results")
                        
                        # Display article title
                        st.markdown(f'<div class="article-title">{article_data["title"]}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Generate and display summary
                        summary = summarize_chain.run(
                            title=article_data["title"],
                            content=article_data["content"]
                        )
                        
                        # Display statistics
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.markdown('<div class="stats-box">📝 Original Length<br/>'
                                      f'<strong>{len(article_data["content"])} chars</strong></div>', 
                                      unsafe_allow_html=True)
                        with col_stats2:
                            st.markdown('<div class="stats-box">⚡ Summary Length<br/>'
                                      f'<strong>{len(summary)} chars</strong></div>', 
                                      unsafe_allow_html=True)
                        with col_stats3:
                            reduction = round((1 - len(summary)/len(article_data["content"])) * 100)
                            st.markdown('<div class="stats-box">📊 Reduction<br/>'
                                      f'<strong>{reduction}%</strong></div>', 
                                      unsafe_allow_html=True)
                        
                        # Display summary
                        st.markdown("#### Generated Summary")
                        st.markdown(summary)
                        
                        # Add download button
                        st.download_button(
                            "Download Summary",
                            summary,
                            file_name="article_summary.txt",
                            mime="text/plain"
                        )
        else:
            st.warning("Please enter a valid URL.")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        Built with ❤️ using Streamlit and LangChain
    </div>
    """, 
    unsafe_allow_html=True
)
