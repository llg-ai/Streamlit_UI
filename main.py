from pathlib import Path
import streamlit as st
import pdfplumber
import argparse
import json
from argparse import RawTextHelpFormatter
import requests
from typing import Optional
import warnings
import streamlit_survey as ss
import random

try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = "https://easonchen19-llg-ai-workflow.hf.space"
FLOW_ID = "ce5141a7-b61a-4d72-b997-c18057307a5c"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings
HF_API_KEY = st.secrets["hf_api_key"]

st.set_page_config(
    page_title="LLG",
    page_icon="💵",
)

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")

# Load Animation
# animation_symbol = "❄"
animation_symbol = "💵"
animation_symbol2 = "$"

st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol2}</div>
    <div class="snowflake">{animation_symbol2}</div>
    <div class="snowflake">{animation_symbol2}</div>
    """,
    unsafe_allow_html=True,
)


# st.markdown(
#     "<h1 style='text-align: center;'>Happy Holiday Season</h1>", unsafe_allow_html=True
# )

st.balloons()
# st.snow()


with st.sidebar:
        container2 = st.container
        uploaded_file = st.file_uploader("Upload a pdf to start:", type=["pdf"])
        st.text("OR")
        st.text("Put in company names to start:")
        
        acquirer = st.text_input("Acquirer", "")
        acquired = st.text_input("Company acquired", "")

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "ChatInput-U8YtE": {},
  "ChatOutput-OIwjh": {},
  "ParseData-KhOtb": {},
  "Prompt-atqt2": {
      "template": "After users input a file or some data, you should help users summarize it in high-level, and also return the relative link from sec.gov website\nThe high-level information includes key takeaways, like: termination fee, deadline, important date and other important numbers that users should know.\n\nAfter that, users typically will ask you some questions in the document below, and can you also answer their questions in simple 1 sentence or 2. \n\nBe careful there might be more than 2 or 3 termination fees, you should return each and all of them and summarize the corresponding scenarios. \n\n\n\n---\n\n{Document}\n\n---\n\n\nQuestion:\n\nAlso, return the context where you find the information and list them below, like a few sentences length?\n\nwhen you answer question, can you also link the relative announcement you found in sec.gov website? i meant the merger or M&A announcement link in sec government website as well as other relative links or news."
  }, 
  "OpenAIModel-4ybSU": {
    "api_key": "openai_api_key"
  },
  "APIRequest-uQLqa": {
    "body": [],
    "headers": [],
    "urls": [
      "https://mvp-fastapi.onrender.com/"
    ]
  },
  "Prompt-mzxbB": {
    "template": "After users input a file or some data, you should help users summarize it in high-level, and also return the relative link from sec.gov website\nThe high-level information includes key takeaways, like: termination fee, deadline, important date and other important numbers that users should know.\n\nAfter that, users typically will ask you some questions in the {search_results}{input_value} below, and can you also answer their questions in simple 1 sentence or 2. \n\nBe careful there might be more than 2 or 3 termination fees, you should return each and all of them and summarize the corresponding scenarios. \n\n\n\n---\n\n{search_results}{input_value}\n\n---\n\n\nQuestion:\n\nAlso, return the context where you find the information and list them below, like a few sentences length?\n\nwhen you answer question, can you also link the relative announcement you found in sec.gov website? i meant the merger or M&A announcement link in sec government website as well as other relative links or news. Also, remember - if there are some value in {input_value}, do some online search before returning to users. When there are data about different mergers in the {search_results}and {input_value}, you should return them separately, then ask users to let them decide which data to return"
  },
  "TextInput-ySsT3": {
    "input_value": acquirer + "'s acquisition of " + acquired
  }
}

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="""Run a flow with a given message and optional tweaks.
Run it like: python <your file>.py "your message here" --endpoint "your_endpoint" --tweaks '{"key": "value"}'""",
        formatter_class=RawTextHelpFormatter)
    # parser.add_argument("message", type=str, help="The message to send to the flow")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=json.dumps(TWEAKS))

    args = parser.parse_args()
    try:
      tweaks = json.loads(args.tweaks)
    except json.JSONDecodeError:
      raise ValueError("Invalid tweaks JSON string")

    hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            div.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
            </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    # Streamlit UI
    st.title(f":balloon: :green[LLG AI]")
    st.markdown(f":green[AI-powered information retrieval in finance!]")

    css="""
    <style>
        [data-testid="stSidebarContent"] {
            background-image: linear-gradient(to right, #0eff00, #0a5d00);
        }
        [data-testid="stFileUploaderDropzone"] {
            # background: #000000; 
            height: 90%;
            width: 85%;
            # background-image: linear-gradient(to left, #0eff00, #0a5d00);
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
            # background: #000000; 
            # background-image: linear-gradient(to left, #0eff00, #0a5d00);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # update the file count
    url_count = "https://mvp-fastapi.onrender.com/count"
    count = requests.request("GET", url_count).json()["count"][0]
    # count += 1
    with st.sidebar:
        st.metric(label="Questions answered based on files:", value=count, border=True)

    if uploaded_file is not None:
        # update file count
        count += 1
        data = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for p in pdf.pages:
                # for image in p.images:
                #     print(image["page_number"]) # until here, everything works!
                #     with open(f"image_{image['page_number']}.jpg", "wb") as f:
                #         print(type(image["stream"]))
                
                data += p.extract_text()

        # this is the url for fastapi
        url = "https://mvp-fastapi.onrender.com/"

        # POST request
        payload = {
            "item": data,
            "count": count
        }
       
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        
        # Display user message in chat message container
        with st.chat_message("user"):               
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            
            res = run_flow(
                message=prompt,
                endpoint=args.endpoint,
                output_type="chat",
                input_type="chat",
                tweaks=tweaks,
                api_key=HF_API_KEY
            )

            response = res["outputs"][0]["outputs"][0]["results"]["message"]["data"]["text"]
            # Add response to chat history
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
       
if __name__ == "__main__":
    main()
