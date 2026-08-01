#========STEP 1=========
import time
import base64
from urllib.parse import quote
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

    _last_image_call_time = [0.0]

    def generate_image(img_prompt: str) -> str:
        """Generates an image for the given prompt and returns a data: URI
        (base64-encoded image) that can be used directly as the src of an
        <img> tag in the generated HTML. Do NOT reference a local file path.
        The image is fetched server-side, so just use the returned string
        as-is in the src attribute — no extra fallback handling needed."""
        # Pollinations' anonymous tier allows ~1 image request per 15s.
        # Space out calls so images actually generate instead of erroring.
        elapsed = time.time() - _last_image_call_time[0]
        if elapsed < 15:
            time.sleep(15 - elapsed)

        encoded_prompt = quote(img_prompt)
        seed = abs(hash(img_prompt)) % 100000
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width=1024&height=576&nologo=true&seed={seed}&referrer=ppt_builder_app"
        )
        try:
            resp = r.get(url, timeout=60)
            _last_image_call_time[0] = time.time()
            content_type = resp.headers.get("content-type", "")
            if resp.status_code == 200 and content_type.startswith("image"):
                b64 = base64.b64encode(resp.content).decode("utf-8")
                return f"data:{content_type};base64,{b64}"
        except Exception:
            _last_image_call_time[0] = time.time()

        # Only reached if generation genuinely failed (network/timeout/etc.)
        return f"https://picsum.photos/seed/{seed}/1024/576"

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
        prompt = """Based on the query below, build a visually bold slide
        deck as a single HTML file, styled like an actual PowerPoint/Keynote
        presentation — not a plain document. Use the generate_image tool for
        any slide that needs a visual, and use search_latest_info if current
        or factual information is required.

        Design requirements (do not skip these):
        - The page background must be a bold, colorful gradient (2-3 colors)
          that fits the topic's mood — NOT plain white or light grey.
        - Each slide is a full-width "card" section (min-height: 90vh,
          scroll-snap-align: start on a scroll-snap-type: y mandatory
          container) so it feels like one slide per screen, like real slides.
        - Give the very first slide a distinct "title slide" treatment: large
          centered title, subtitle, no body image, on the boldest part of the
          gradient.
        - Every content slide needs a strong accent color (used for headings,
          borders, or highlight bars) that contrasts against the gradient
          background, plus a semi-opaque/glass card (background: rgba(...))
          so text stays readable over the gradient.
        - Include a <style> block with real CSS: import a Google Font for
          headings (e.g. Poppins/Montserrat) and a different one for body
          text, and use consistent spacing, border-radius, and box-shadow.
        - For every image, call the generate_image tool and use the exact
          returned string as the src of an <img> tag (it will be a data URI —
          use it as-is). Style images with fixed width, border-radius, and
          object-fit: cover. Do NOT reference a local file path or filename
          for images, and do not add onerror handlers — the tool already
          guarantees a usable image.

        Output ONLY the final HTML (including the <style> block), no markdown,
        no explanation before or after it.

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
