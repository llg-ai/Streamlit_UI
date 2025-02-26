import streamlit_survey as ss
import streamlit as st
import json
import pygsheets
import pandas as pd

#authorization
# gc = pygsheets.authorize(service_file='./llg-survey.json')

# Create empty dataframe
# df = pd.DataFrame()

# Create a column
# df['name'] = ['John', 'Steve', 'Sarah']
#open the google spreadsheet (where 'LLG_Survey' is the name of my sheet)
# sh = gc.open('LLG_Survey')
#select the first sheet 
# wks = sh[0]

survey = ss.StreamlitSurvey()

# Streamlit UI
st.title("Feedback of LLG")
st.text("")


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
    
    # df = pd.read_json(f'[{data}]')
    # st.write(df)
    #update the first sheet with df
    # current_df = wks.get_as_df()
    # num_rows = current_df.shape[0]
    # num_cols = current_df.shape[1]
    
    # wks.set_dataframe(df,(num_rows+2, 1))

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
    