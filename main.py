import streamlit as st
import requests
import pdfplumber
from model_langflow import run_flow
import json

FLOW_ID = "a4e1a2fb-3e8a-4fa8-a99c-ec7bb47e94c1"
APPLICATION_TOKEN = "AstraCS:NoHkztKImvlyspGlzlHuvRzn:f6cb16f4f422734ac45c0b7c90df3dae7cfb7e5c428c9efb1fad9d5037676eed"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

def main():
   # Streamlit UI
    st.title("LLG-AI Chatbot")
    
    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf", "docx"])

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
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            res = run_flow(
                message=prompt,
                endpoint=FLOW_ID,
                application_token=APPLICATION_TOKEN
                )
            # print("res: ", res)
            response = st.write(res["outputs"][0]["outputs"][0]["results"]["message"]["data"]["text"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
