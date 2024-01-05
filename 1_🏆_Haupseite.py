'''
    Autor: Christoph Struth
    Datum: 10.02.2023

    Haupseite des Yugioh Streamlit Dashboard, die App ist auf mehrere Seiten ausgelegt,
    die unabhänging voneinander interagieren und verschiedene Statistiken zu dem Elosystem
    anzeigen.
    Diese Hauptseite zeigt die globalen Statistiken zu den Spielen und Spieler, sowie die
    einzelnen Deckkategorien
'''
# page classes
from pages._pages.FrontPage import FrontPage
import pages._pages.layouts as layouts
# data models
from datamodel import datamodel

def run():
    # get data
    dm = datamodel()
    df, cols, hist_cols = dm.get()
    tdf = dm.get_tournament_data()
    FrontPage(df.copy(), tdf, hist_cols)

if __name__ == "__main__":
   layouts.insert_layout_htmls()
   layouts.insert_metric_style()
   run()