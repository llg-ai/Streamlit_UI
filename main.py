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
import ollama
import re
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(
    page_title="LLG",
    page_icon="💵",
)

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = st.secrets["OPEN_API_KEY"]

# setup chat model, embedding and vector_store
llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)


# if not os.getenv("DEEPSEEK_API_KEY"):
#     os.environ["DEEPSEEK_API_KEY"] = "sk-79d2f50d21a34b8fbe45b17e2d205d8e"

from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

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
                
                from langchain_community.document_loaders.parsers import LLMImageBlobParser
                from langchain_openai import ChatOpenAI
                loader = PyPDFLoader(
                    uploaded_file.name,
                    mode="page",
                    images_inner_format="markdown-img",
                    images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
                )
                pages = loader.load()

                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_openai.embeddings import OpenAIEmbeddings

                text_splitter = SemanticChunker(
                    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
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
            
            # from langchain import hub
            # rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

            # # build a retriever from vector_store
            # retriever = vector_store.as_retriever(
            #     search_type="mmr",
            #     search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            # )

            # from operator import itemgetter

            # from langchain_core.output_parsers import StrOutputParser
            # from langchain_core.prompts import ChatPromptTemplate
            # from langchain_core.runnables import Runnable, RunnablePassthrough, chain

            # contextualize_instructions = """Convert the latest user question into a standalone question given the chat history. Don't answer the question, return the question and nothing else (no descriptive text)."""
            # contextualize_prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", contextualize_instructions),
            #         ("placeholder", "{chat_history}"),
            #         ("human", "{question}"),
            #     ]
            # )
            # contextualize_question = contextualize_prompt | llm | StrOutputParser()

            # qa_instructions = (
            #     """Answer the user question given the following context:\n\n{context}."""
            # )
            # qa_prompt = ChatPromptTemplate.from_messages(
            #     [("system", qa_instructions), ("human", "{question}")]
            # )


            # build a chain of chain
            # question_answer_chain = create_stuff_documents_chain(llm, prompt_tem)
            # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            # invoke the chain
            # response = rag_chain.invoke({"input": prompt})


            # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            # conversation_chain = ConversationalRetrievalChain.from_llm(
            #     llm=llm,
            #     chain_type="stuff",
            #     retriever=vector_store.as_retriever(),
            #     memory=memory
            # )

            # query = prompt
            # result = conversation_chain({"question": query})
            # answer = result["answer"]

            embedding = embeddings.embed_query(prompt)

            results = vector_store.similarity_search_by_vector(embedding)
            # for result in results:
            #     st.markdown(result)
            #     st.markdown("---------------------------------------------------------")

            from langchain_core.prompts import ChatPromptTemplate

            prompt_cus = ChatPromptTemplate(
                [
                    (
                        "system",
                            """

                            When users ask for high level summary or key takeaways, you shou search entire file and collect important memos for users
                            Including:
                            - Acquisition Details
                            - Financing Structure
                            - Termination Clauses
                            - Voting Agreement
                            - Risks & Forward-Looking Statements
                            - Press Release & Investor Meetings
                            - Due Date/Announcement Date

                            Be sure also return the SEC filing link to users. 

                            When users ask a specific question, like termination fee, you should help users answer questions 
                            based on the context: {context}. If you do not know, just "say I do not know!" Do not make up answers! 
                            You should also return the underlying context where you found the answer for the question.

                            """
                    ),
                    ("human", "{input}"),
                ]
            )

            chain = prompt_cus | llm
            res = chain.invoke(
                {
                    "context": results,
                    "input": prompt,
                }
            )
            response = res.content
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
       
if __name__ == "__main__":
    main()
