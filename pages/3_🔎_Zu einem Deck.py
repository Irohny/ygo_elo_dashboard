# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.DetailedInfoPage import DetailedInfoPage
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    st.title('Yu-Gi-Oh! Elo-Dashbord')
    DetailedInfoPage(df.copy(), hist_cols)

if __name__ == "__main__":
   run()