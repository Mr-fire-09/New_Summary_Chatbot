from langchain_google_genai import ChatGoogleGenerativeAI  # Fixed typo
import os
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate  # Fixed typo in 'promts'
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)  # Fixed typo: 'Genrative'

def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        return f"Error extracting news from {url}: {e}"


summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]  # Fixed placeholder
)

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

def summarize_news(url):
    print(f"\nFetching news from: {url}")
    article = extract_news(url)
    if article.startswith("Error extracting"):
        return article

    summary = summarize_chain.run(article=article)  # Fixed `-` to `=`
    return summary


if __name__ == "__main__":  # Fixed `_name_` to `__name__`
    user_url = input("Please enter your URL:\n")
    result = summarize_news(user_url)
    print(f"\nSummary:\n{result}")
