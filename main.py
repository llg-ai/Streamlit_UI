import streamlit as st
import json
import requests
import streamlit_survey as ss
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_deepseek import ChatDeepSeek
import pypdf
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker

st.set_page_config(
    page_title="LLG",
    page_icon="💵",
)

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]

# setup chat model, embedding and vector_store
llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=st.secrets["deepseek_api_key"]
#     # other params...
# )

# prompt_cus = ChatPromptTemplate(
#                 [
#                     (
#                         "system",
#                             """

#                             When users ask for high level summary or key takeaways, you shou search entire file and collect important memos for users
#                             Including:
#                             - Acquisition Details
#                             - Financing Structure
#                             - Termination Fee
#                             - Voting Agreement
#                             - Risks & Forward-Looking Statements
#                             - Press Release & Investor Meetings
#                             - Due Date/Announcement Date

#                             Be sure also return the SEC filing link to users. 

#                             When users ask a specific question, like termination fee, you should help users answer questions 
#                             based on the context: {context}. If you do not know, just "say I do not know!" Do not make up answers! 
#                             You should also return the underlying context where you found the answer for the question.

#                             """
#                     ),
#                     ("human", "{input}"),
#                 ]
# )

# chain = prompt_cus | llm

template2 = """
    Answer users questions {question}!

    Before answer the question, you should think about the question
    is about high level summary or key takeaways
    OR
    is about a specific question.

    If you believe that this is a high level summary or takeaways question, you should search entire file and collect important memos for users
    Including:
    - Acquisition Details
    - Financing Structure
    - Termination Clauses
    - Voting Agreement
    - Risks & Forward-Looking Statements
    - Press Release & Investor Meetings
    - Due Date/Announcement Date

    Be sure also return the SEC filing link to users. 

    If you think this is a specific question,like termination fee, due date valuation and other question you think 
    should fall into this category, you should help users answer questions 
    based on the context: {context}. You should also return the underlying context where you found the answer for the question.
    Also, you should highlight the "keywords".

    If you do not know, just "say I do not know!" Do not make up the answers! 
    
"""

custom_rag_prompt = PromptTemplate.from_template(template2)

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

# Load Animation
# animation_symbol = "❄"
animation_symbol = "🎉" 
animation_symbol2 = "💵"
animation_symbol3 = "$"

st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol2}</div>
    <div class="snowflake">{animation_symbol2}</div>
    <div class="snowflake">{animation_symbol2}</div>
    <div class="snowflake">{animation_symbol3}</div>
    <div class="snowflake">{animation_symbol3}</div>
    <div class="snowflake">{animation_symbol3}</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
        container2 = st.container
        uploaded_file = st.file_uploader("Upload a pdf to start:", type=["pdf"])
        st.text("OR")
        st.text("Put in company names to start:")
        
        acquirer = st.text_input("Acquirer", "")
        acquired = st.text_input("Company acquired", "")

def main():
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
    st.markdown("""<span style="color: green;">AI-powered information retrieval in finance!</span>""", unsafe_allow_html=True)

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
        url = "https://mvp-fastapi.onrender.com/"
        # POST request
        payload = {
            "item": "",
            "count": count
        }
       
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    
        with st.spinner("Think...", show_time=True):

            with open(uploaded_file.name, mode='wb') as w:
                w.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(
                    uploaded_file.name,
                    mode="page",
                    images_inner_format="markdown-img",
                    images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
                )
                pages = loader.load()

                # text_splitter = SemanticChunker(
                #     OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
                # )
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,  # chunk size (characters)
                    chunk_overlap=750,  # chunk overlap (characters)
                    add_start_index=True,  # track index in original document
                )

                all_splits = text_splitter.split_documents(pages)
 
                # Load all splits into VDB
                vector_store.add_documents(documents=all_splits)

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
            # with st.spinner(text = "thinking..."):              
                st.markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

            embedding = embeddings.embed_query(prompt)

            results = vector_store.similarity_search_by_vector(embedding, k=3)

            # pass search result and question into the model chain
            # res = chain.invoke(
            #     {
            #         "context": results,
            #         "input": prompt,
            #     }
            # )

            # response = res.content
            # st.markdown(response)
            
            # build prompt based on the search result
            messages = custom_rag_prompt.invoke({"question": prompt, "context": results})
            # load the prompt into LLM
            response = llm.stream(messages)
            res = st.write_stream(response)

        st.session_state.messages.append({"role": "assistant", "content": res})
       
if __name__ == "__main__":
    main()
