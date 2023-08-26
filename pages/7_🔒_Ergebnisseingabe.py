'''
    Autor: Christoph Struth
    Datum: 10.02.2023
    Fünfte Unterseite des Yugioh Streamlit Dashboards
    Diese Seite beinhaltet die Möglichkeit Ergenbissse von Spiele einzutragen
    um die Elo-Statistiken up zu daten. Zusätzlich können neue Decks angelegt werden 
    oder die Stats einzelner Decks angepasst werden. Backup möglichkeit auch gegeben.
    Diese Seite ist Passwort geschützt um die unkontrollierte Eingabe von außen zu minimieren.
'''
# liberies and frameworks
import streamlit as st
# page classes
from pages._pages.InputPage import InputPage
from pages._pages.layouts import insert_layout_htmls
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()
    # create opening page and all sidetabs 
    #st.title('Yu-Gi-Oh! Elo-Dashbord')
    InputPage(dm, df, hist_cols, cols)

if __name__ == "__main__":
   insert_layout_htmls()
   run()