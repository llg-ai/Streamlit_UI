import streamlit_survey as ss
import streamlit as st
import json
import pygsheets
import pandas as pd
from streamlit_gsheets import GSheetsConnection

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(worksheet="LLG_Survey")

survey = ss.StreamlitSurvey()

# Streamlit UI
st.title("Feedback of LLG")
st.text("")

st.snow()

survey.text_input("What is your name? ")

survey.text_input("What is your email address? ")

survey.radio("How satisfied are you with our product?", options=["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"], horizontal=True)

survey.radio("How easy is it to use our product?", options=["Very easy", "Somewhat easy", "Neutral", "Somewhat difficult", "Very difficult"], horizontal=True)

survey.radio("Will you recommend it to your friends??", options=["Yes, absolutely!", "No, I want to see more features.", "Uncertainly"])

survey.text_input("Is there something else you want us to know? (It could be anything)")

clickSubmit = st.button("Submit")
if clickSubmit == True:
    st.markdown('<h3>Thank you for your feedback!</h3>', unsafe_allow_html=True)
    data = survey.to_json()
    
    new_df = pd.read_json(f'[{data}]')

    for i in range(6):
        new_df.iloc[0][i] = new_df.iloc[0][i]["value"] 
    
    current_df = conn.read(usecols=[0,1,2,3,4,5])
    if current_df is not None:
        updated_df = pd.concat([current_df, new_df])
    else:
        updated_df = current_df

    conn.update(data=updated_df)

css="""
    <style>
        [data-testid="stSidebarContent"] {
            background-image: linear-gradient(to right, #0eff00, #0a5d00);
        }
        [data-testid="stMainBlockContainer"] {
            # background: #ffffff; 
        }
    </style>
    """
st.write(css, unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            div.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
