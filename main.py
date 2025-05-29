from langchain_google_genai import ChatGoogleGenrativeAI
import os 
import requests
from bs4  import BeautifulSoup
from langchain.prompts import PromptTemplate  # ✅ Correct

from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()

llm = ChatGoogleGenrativeAI(model="gemini-1.5-flash",temperature=0.7)


def extract 