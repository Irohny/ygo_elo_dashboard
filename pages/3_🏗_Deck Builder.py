'''
    Autor: Christoph Struth
    Datum: 10.02.2023
    Dritte Unterseite des Yugioh Streamlit Dashboards
    Diese Seite Hilft die Ratios innerhalb eines Decks durch Draw Wahrscheinlichkeiten zu optiomieren
'''
# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.DeckBuilderPage import DeckBuilderPage
import pages._pages.layouts as layouts
# data models
from datamodel import datamodel

def run():
    # get data
    #dm = datamodel()
    #df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    #st.title('Yu-Gi-Oh! Elo-Dashbord')
    DeckBuilderPage()

if __name__ == "__main__":
   layouts.insert_layout_htmls()
   layouts.insert_metric_style()
   run()