#========STEP 1=========
import time
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
import requests as r
import streamlit as st

st.set_page_config(layout="wide")

#=================STEP 2 LOAD ENV and API KEYS==================
st.title("Agentic PPT Generator")
st.header("""User can generate, PPT, Images, and fetch Latest news""")

st.sidebar.title("Give API KEYS")

GOOGLE_API_KEY = st.sidebar.text_input("GOOGLE_API_KEY", type="password")
GROQ_API_KEY = st.sidebar.text_input("GROQ_API_KEY (optional fallback)", type="password")
TAVILY_API_KEY = st.sidebar.text_input("TAVILY_API_KEY", type="password")

leader_agent = None
fallback_agent = None

# Tavily is always required. At least one of Google/Groq is required.
if not TAVILY_API_KEY:
    st.sidebar.error("Tavily API key is required.")
    st.markdown("Get Tavily API key: https://app.tavily.com/playground")

elif not GOOGLE_API_KEY and not GROQ_API_KEY:
    st.sidebar.error("Provide at least one of Google or Groq API key.")
    st.markdown("Get Google API key: https://aistudio.google.com/apikey")

else:
    st.success("API KEYS LOADED")

    model = None
    if GOOGLE_API_KEY:
        options = [
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite",
        ]
        selected_model = st.selectbox("Select Gemini Model", options=options)
        model = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=GOOGLE_API_KEY,
        )

    fallback_model = None
    if GROQ_API_KEY:
        fallback_model = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
        )

    # If only Groq was provided, use it as the primary model instead
    if model is None:
        model = fallback_model
        fallback_model = None

    #==========================STEP 3=============================
    def search_latest_info(query):
        """This function helps to give
        latest search using Tavily
        based on given user query related research or
        contents"""
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query)
        return response

    def generate_image(img_prompt: str) -> str:
        """Generates an image for the given prompt and returns a direct,
        publicly-accessible image URL. Use this URL directly as the src
        of an <img> tag in the generated HTML — do NOT reference a local
        file path, since the browser cannot access files on the server."""
        url = f"https://image.pollinations.ai/prompt/{img_prompt}"
        return url

    leader_agent = create_agent(model=model, tools=[search_latest_info, generate_image])
    if fallback_model is not None:
        fallback_agent = create_agent(model=fallback_model, tools=[search_latest_info, generate_image])

    def invoke_with_fallback(prompt_content):
        """Try the primary agent first; if it errors (auth, quota, deprecated
        model, etc.) retry with the fallback agent if one is configured."""
        try:
            return leader_agent.invoke({'messages': [{'role': 'user', 'content': prompt_content}]})
        except Exception as primary_error:
            if fallback_agent is not None:
                st.warning(f"Primary model failed ({primary_error}). Retrying with fallback model...")
                return fallback_agent.invoke({'messages': [{'role': 'user', 'content': prompt_content}]})
            raise

    def extract_text(response):
        """Gemini returns content as a list of blocks (content[-1]['text']),
        but Groq returns content as a plain string. Handle both safely."""
        content = response['messages'][-1].content
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            last = content[-1]
            if isinstance(last, dict):
                return last.get('text', str(last))
            return str(last)
        return str(content)

    def run_agent(query):
        prompt = """Based on Below given Query,
        your task is to call specific tool, first to
        promptify user prompt, then call the generate_image tool for each
        slide that needs a visual, or the search_latest_info tool if
        latest information is required. Give slide dynamic, ui ux,
        with creative design.

        IMPORTANT: For every image, you MUST call the generate_image tool
        and use the exact URL it returns as the src attribute of an <img>
        tag (e.g. <img src="THE_RETURNED_URL">). Do NOT reference a local
        file path or filename for images — only the URL returned by the tool.

        Give Final response output in HTML, no markdowns
        user query given below:""" + query

        response = invoke_with_fallback(prompt)
        code = extract_text(response)
        return code

    # ============ Step 4 STREAMLIT NAVBARS ============
    tab1, tab2, tab3 = st.tabs([
        "Generate Image",
        "Fetch News",
        "Generate PPT",
    ])

    user_input = st.text_area("Write Prompt & click Enter")

    if user_input:
        with tab1:
            if st.button("Click to Generate Image", key="Image-Button"):
                with st.spinner("Running Agent"):
                    try:
                        url = generate_image(user_input)
                        st.image(url)
                    except Exception as err:
                        st.error(f"Error Code: {err}")

        with tab2:
            if st.button("Fetch Latest News", key="News-Button"):
                with st.spinner("Running Agent"):
                    try:
                        prompt = """Give Latest News Related to Given user Query
                        in Dynamic HTML, Output with cards Design Format.
                        Strict HTML Output, No Any markdowns Response
                        User Query: """ + user_input

                        response = invoke_with_fallback(prompt)
                        code = extract_text(response)
                        st.html(code, width="stretch", unsafe_allow_javascript=True)
                    except Exception as err:
                        st.error(f"Error Code: {err}")

        with tab3:
            if st.button("Click To Generate PPT", key="PPT-Button"):
                with st.spinner("Running Agent"):
                    try:
                        code = run_agent(user_input)
                        st.html(code, width="stretch", unsafe_allow_javascript=True)

                        st.download_button(
                            label="DOWNLOAD PPT",
                            data=code,
                            file_name="ppt.html",
                            mime="text/html",
                        )
                    except Exception as err:
                        st.error(f"Error Code: {err}")
    else:
        st.info("Enter a prompt above to get started.")
