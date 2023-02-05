# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.FrontPage import FrontPage
from pages._pages.layouts import insert_layout_htmls
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    st.title('Yu-Gi-Oh! Elo-Dashbord')
    st.sidebar.header ('Wähle eine Seite')
    FrontPage(df.copy(), hist_cols)

if __name__ == "__main__":
   insert_layout_htmls()
   run()