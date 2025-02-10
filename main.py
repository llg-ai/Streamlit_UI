import streamlit as st
import requests
import pdfplumber
# from deployment.Streamlit_UI.revoke_langflow import run_flow
import json
from typing import Optional

# This is the base URL for the Langflow API
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "cb683101-5ceb-4e35-9978-d402fc72e89d"
FLOW_ID = "a4e1a2fb-3e8a-4fa8-a99c-ec7bb47e94c1"
APPLICATION_TOKEN = st.secrets["langflow_api_token"]
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

def run_flow(message: str,
  endpoint: str,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  application_token: Optional[str] = None) -> dict:
  
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if application_token:
        headers = {"Authorization": "Bearer " + application_token, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def main():

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            div.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    # Streamlit UI
    st.title(":balloon: LLG AI")
    st.text("AI-powered information retrieval in finance!")

    with st.sidebar:
        container2 = st.container
        uploaded_file = st.file_uploader("Upload a pdf to start:", type=["pdf"])
        st.text("OR")
        st.text("Put in company names to start:")
        
        st.text_input("Acquirer", "")
        st.text_input("Company acquired", "")

    if uploaded_file is not None:
        
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
            "item": data
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
                endpoint=FLOW_ID,
                application_token=APPLICATION_TOKEN
            )

            response = res["outputs"][0]["outputs"][0]["results"]["message"]["data"]["text"]
            # Add response to chat history
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
       

if __name__ == "__main__":
    main()
