'''
    Autor: Christoph Struth
    Datum: 10.02.2023
    Erste Uterseite des Streamlit Yugioh Dashboards.
    Diese Seite kann dafür genutzt werden die Eloverläufe und die Statistiken
    mehrerer Decks miteinander zu vergleichen
'''
# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.DeckComparisonPage import DeckComparisionPage
import pages._pages.layouts as layouts
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    #sst.title('Yu-Gi-Oh! Elo-Dashbord')
    DeckComparisionPage(df.copy(), hist_cols)

if __name__ == "__main__":
   layouts.insert_layout_htmls()
   layouts.insert_metric_style()
   run()