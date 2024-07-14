import streamlit as st
import pandas as pd

from DataModel.DeckModel import DeckModel
from DataModel.TournamentModel import TournamentModel

def load_data()->None:
    # intialization of session sate
    if 'login' not in st.session_state:
        st.session_state['sorting'] = ['Elo-Rating']
        st.session_state['tournament'] = 'Alle'
        st.session_state['deck_i'] = ''
        st.session_state['deck'] = []
        st.session_state['reload_flag'] = False
        st.session_state['login'] = None
        
    # Deck Data
    dm = DeckModel()
    df, cols, hist_cols = dm.get()
    st.session_state['columns'] = cols
    st.session_state['history_columns'] = hist_cols
    st.session_state['owner'] = df['Owner'].unique() 
    st.session_state['types_select'] = df['Type'].unique()
    st.session_state['tier'] = df['Tier'].unique()
    								
    # Torunament Data
    dm = TournamentModel()
    st.session_state['tournament_data'], st.session_state['torunament_aggregated'], df_Score = dm.get()
    st.session_state['deck_data'] = pd.merge(df, df_Score, on='Deck', how='left')
    st.session_state['deck_data'].fillna(0)