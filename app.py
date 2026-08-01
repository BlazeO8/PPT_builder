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

# Search latest info using Tavily

def search_latest_info(query):
    """
    This function helps to give
    latest search using Tavily
    based on given user query related research or
    contents
    """

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query)
    return response

def generate_image(img_prompt, slide_no=1):
    """This function helps user to generate
    image using free api, with given
    img_prompt, with slide no"""

    url =  f"https://image.pollinations.ai/{img_prompt}"

    import requests as r
    content = r.get(url).content
    with open(f"ai_image_{slide_no}.jpeg", "wb") as f:
        f.write(content)
    return url

def run_agent(leader_agent, query):
    prompt = f"""Based on Below given Query,
    your task is to call specific tool, first to
    promptify user prompt, than call image tool, or
    latest search if required. give slide dynamic, ui ux,
    with creative design, keep help of function to generate image
    based on given topic,
    Generate image using
    with number of slide asked, and use time sleep to hit image request on server
    and using file handling embed this in output html, use java script function
    give Final response output in HTML, no markdowns
    user query given below:"""

    prompt
        ]
    })

    code = response["messages"][-1].content[-1]["text"]
    return code

# leader_agent creation
if all(all_API)    
  leader_agent = create_agent(
      model=model,
      tools=[
          search_latest_info,
          # generate_image])

  leader_agent
else:
    st.info("Give API-Keys first to load Agent")
