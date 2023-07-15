'''
    Autor: Christoph Struth
    Datum: 10.02.2023
    Zweite Unterscheite des Yugioh Streamlit Dashboards
    Diese Seite zeigt alle wichtigen Statistiken zu einem einzelnen Deck an.
'''
# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.DetailedInfoPage import DetailedInfoPage
import pages._pages.layouts as layouts
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    #st.title('Yu-Gi-Oh! Elo-Dashbord')
    DetailedInfoPage(df.copy(), hist_cols)

if __name__ == "__main__":
   layouts.insert_layout_htmls()
   layouts.insert_metric_style()
   run()