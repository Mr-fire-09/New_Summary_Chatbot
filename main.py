from langchain_google_genai import ChatGoogleGenrativeAI
import os 
import requests
from bs4  import BeautifulSoup
from langchain.prompts import PromptTemplate  # ✅ Correct

from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

llm = ChatGoogleGenrativeAI(model="gemini-1.5-flash",temperature=0.7)


def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting news from {url}: {e}")
        return None