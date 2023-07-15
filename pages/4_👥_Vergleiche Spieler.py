'''
    Autor: Christoph Struth
    Datum: 10.02.2023
    Dritte Unterseite des Yugioh Streamlit Dashboards
    Diese Seite zeigt die Statistiken aller Spieler in dem System.
'''
# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.PlayerPage import PlayerPage
import pages._pages.layouts as layouts
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    #st.title('Yu-Gi-Oh! Elo-Dashbord')
    PlayerPage(df.copy(), hist_cols)

if __name__ == "__main__":
   layouts.insert_layout_htmls()
   layouts.insert_metric_style()
   run()