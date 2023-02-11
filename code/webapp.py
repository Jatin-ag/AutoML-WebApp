#General imports
import streamlit as st
import pandas as pd
import os

#Profiling CapabilitY imports
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

#Machine Learning imports
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://img.freepik.com/free-vector/robotic-artificial-intelligence-technology-smart-lerning-from-bigdata_1150-48136.jpg?size=626&ext=jpg")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application allows you to build an automated machine learning pipeline using Streamlit, Pandas Profiling and PyCaret")

if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv", index_col = None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("source_data.csv")
        st.dataframe(df)
        

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "ML":
    st.title("Machine Learning")
    target = st.selectbox("Select Your Target (Value you wish to predict)", df.columns)
    if st.button("Train Models"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")


if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
         st.download_button("Download the Model", f, "trained_model.pkl" )