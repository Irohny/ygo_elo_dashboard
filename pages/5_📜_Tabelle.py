'''
    Autor: Christoph Struth
    Datum: 10.02.2023
    Vierte Unterseite des Yugioh Streamlit Dashboards
    Diese Seite zeigt die aktuelle Tabelle der Elo Punktzahl aller Decks.
    Diese Seite beinhaltet mehrere Filtermöglichkeiten um Tabellen auf
    einzelne Spieler oder Deckklassen einzustellen.
'''
# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.TablePage import TablePage
from pages._pages.layouts import insert_layout_htmls
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()

    # create opening page and all sidetabs 
    #st.title('Yu-Gi-Oh! Elo-Dashbord')
    TablePage(df.copy())

if __name__ == "__main__":
   insert_layout_htmls()
   run()