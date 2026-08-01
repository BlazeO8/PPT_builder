#========STEP 1=========
import os
import time
import langchain
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import pytesseract as pyt
from tavily import TavilyClient
import numpy as np
import streamlit as st

#=================STEP 2 LOAD ENV and API KEYS==================
st.title("Agentic  PPT Generator")
st.header("""User can generate,PPT,Images,and fetch Latest news""")

st.sidebar.title("Give API KEYS")

GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
TAVILY_API_KEY = st.sidebar.text_input("TAVILY_API_KEY", type="password")

ALL_API = [GOOGLE_API_KEY, TAVILY_API_KEY]

if not all(ALL_API):
    st.sidebar.error("Must Pass All API-Keys")

    url = "https://aistudio.google.com/api-keys"
    st.markdown(f"Get Google API key-{url}")

    url = "https://app.tavily.com/playground"
    st.markdown(f"Get Tavily API key-{url}")

elif all(ALL_API):
    st.success("API KEYS LOADED")

    options = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash"
    ]

    selected_model = st.selectbox("Select-Model", options=options)

    model = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=GOOGLE_API_KEY
    )

else:
    st.sidebar.info("Try Valid API-keys")
