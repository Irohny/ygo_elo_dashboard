# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.TablePage import TablePage
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    st.title('Yu-Gi-Oh! Elo-Dashbord')
    TablePage(df.copy())

if __name__ == "__main__":
   run()