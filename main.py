import streamlit as st

from DataModel.utils.load_data import load_data
from streamlit_navigation_bar import st_navbar
from NavBar import get_pages
from utils.layouts import header_styling

import pages as p

if __name__ == '__main__':
    st.set_page_config(page_title='YGO-Elo-Dashboard', 
                       page_icon=':trophy:',
                       layout='wide',
                       initial_sidebar_state='collapsed')
    if 'deck_data' not in st.session_state:
        load_data()

    page_dict = get_pages()
    page = st_navbar(list(page_dict.keys()), styles=header_styling(),
                     #logo_path='Deck_Icons/ygo_icon.svg', 
                     #logo_page='Hauptseite',
                     options = {"show_menu": True,"show_sidebar": False,}) 
    
    if page == 'Hauptseite':
        p.StatsPage()
    elif page == 'Spieler':
        p.PlayerPage()
    elif page == 'Tabelle':
        p.TablePage()
    elif page == 'Deck Details':
        p.DetailedInfoPage()
    elif page == 'Deck Builder':
        p.DeckBuilderPage()
    elif page == 'Deckvergleich':
        p.DeckComparisionPage()
    elif page == 'Eingabe':
        p.InputPage()
    elif page == 'Login':
        p.LoginPage()