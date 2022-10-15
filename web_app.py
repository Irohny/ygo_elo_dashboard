#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:47:10 2022

@author: christoph
"""

import numpy as np
import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from deta import Deta
from functions import *
import os
from dotenv import load_dotenv

load_dotenv(".env")
key = os.getenv('key')
deta = Deta(key)
db = deta.Base("ygo_elo_database")
st.set_page_config(page_title='YGO-Elo-Dashboard', page_icon=':trophy:' ,layout='wide', initial_sidebar_state='collapsed')
    
# --- USER AUTHENTICATION ---
with st.sidebar:

      names = ['Chriss']
      usernames = ['Chriss']
      
      file_path = Path(__file__).parent / "hashed_pw.pkl"
      with file_path.open("rb") as file:
          hashed_passwords = pickle.load(file)
        
      authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
          'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
      
      name, authentication_status, username = authenticator.login("Login", "main")
      
      if authentication_status:
          authenticator.logout('Logout', 'main')
          st.write(f'Welcome *{name}*')
          secrets = True
      elif authentication_status == False:
          st.error('Username/password is incorrect')
          secrets = False
      elif authentication_status == None:
          st.warning('Please enter your username and password')
          secrets = False
          
          
st.title('Yu-Gi-Oh! Elo-Dashbord')

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# build data
df, hist_cols, save_cols = get_all_data(db)
hist_cols = sort_hist_cols(hist_cols)
df_elo = fetch_and_clean_data(df.copy(), hist_cols)

df_select = df_elo.copy()
cols = df_elo.columns.to_list()
ges_stats, vgl_decks, all_in_one, stats_table, vgl_player, update_spiele = st.tabs(['Gesamtstatistik', 'Deckvergleich', 'Deckinfo', 'Tabelle','Spielervergleich', 'Update neue Spiele'])

if "owner" not in st.session_state:
    st.session_state = {'owner':df_elo['Owner'].unique(), 
                        'types_select':df_elo['Type'].unique(),
                        'tier':df_elo['Tier'].unique(),
                        'sorting':['Elo-Rating'],
                        'tournament':[],
                        'deck_i':'',
                        'deck':[]
                        }
# Page with all statisitics
with ges_stats:
      make_stats_side(df_elo, hist_cols)

# vgl Decks
with vgl_decks:
      st.subheader('Wähle deine Decks')
      decks = st.multiselect(
            '',
             options = df_elo['Deck'].unique(),
             default = st.session_state['deck'])
      
      vergleiche_die_decks(df_elo, hist_cols, decks)

# infos zu einem deck 
with all_in_one:
      deck_ops = df_elo['Deck'].astype(str).to_numpy()
      deck_ops = np.append(np.array(['']), deck_ops)
      deck = st.selectbox('Wähle ein deck', options=deck_ops, index=0)
      alles_zu_einem_deck(deck, df_elo, hist_cols)
                  
# darstellung der tabelle
with stats_table:
      make_stats_table(df_elo)
      
# vgl spieler
with vgl_player:
      vergleiche_die_spieler(df_elo, hist_cols)

with update_spiele:
      if secrets:
            st.header('Trage neue Spielergebnisse ein: ')
            with st.form('Input new Games'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        
                        deck1 = st.selectbox('Wähle Deck', 
                                  options=df_elo['Deck'].unique(),
                                  key='deck1')
                        erg1 = st.number_input('Score ', 
                                               min_value=0,
                                               max_value=10000,
                                               step=1,
                                               key='erg2')
                  
                  with inputs[2]:
                        pass
                        
                  with inputs[3]:
                        deck2 = st.selectbox('Wähle Deck', 
                                  options=df_elo['Deck'].unique(),
                                  key='deck2')
                        
                        erg2 = st.number_input('Score ', 
                                               min_value=0,
                                               max_value=10000,
                                               step=1,
                                               key='erg1')
                        
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              update_elo_ratings(deck1, deck2, erg1, erg2, df_elo, hist_cols, save_cols, db)
                   
            st.header('Trage neuen Turniersieg ein:')
            with st.form('Tourniersieg'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        
                        deck_tour = st.selectbox('Wähle Deck', 
                                  options=df_elo['Deck'].unique(),
                                  key='deck tournament')
                        tournament = st.selectbox('Wähle Turnier:', 
                                                  options=['Wanderpokal', 'Meisterschaft',
                                                           'Liga Pokal', 'Fun Pokal'])
                        
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              update_tournament(deck_tour, tournament, df_elo, save_cols, hist_cols, db)
                  
            
            st.header('Trage neues Deck ein:')
            with st.form('neies_deck'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        new_deck = st.text_input('Names des neuen Decks', key='new deck')
                        owner = st.text_input('Names des Spielers', key='player')
                        deck_type = st.selectbox('Wähle einen Decktype:',
                                                 options=df_elo['Type'].unique(), key='decktype')
                  with inputs[2]:
                        attack = st.number_input('Attack-Rating', min_value=0, max_value=5, step=1, key='attack')
                        control = st.number_input('Control-Rating', min_value=0, max_value=5, step=1, key='control')
                        resilience = st.number_input('Resilience-Rating', min_value=0, max_value=5, step=1, key='resilience')
                        
                  with inputs[3]:
                        recovery = st.number_input('Recovery-Rating', min_value=0, max_value=5, step=1, key='recovery')
                        combo = st.number_input('Combo-Rating', min_value=0, max_value=5, step=1, key='combo')
                        consistency = st.number_input('Consistency-Rating', min_value=0, max_value=5, step=1, key='consistency')
                        
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              insert_new_deck(new_deck, owner, attack, control, recovery, consistency, combo, resilience, deck_type, db)
                              
            st.header('Update History:')
            with st.form('history_update'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        pass
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              update_history(df_elo, hist_cols, save_cols, db)
                              
            st.header('Modfiziere Deckstats:')
            with st.form('Mod Stats'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        deck_choose = st.selectbox('Wähle Deck', 
                                  options=df_elo['Deck'].unique(),
                                  key='deck_modify')
                        
                  with inputs[2]:
                        in_stats = st.selectbox('Wähle Eigenschaft zum verändern:',
                                     options=['Attack', 'Control', 	'Recovery',
                                              'Consistensy',	 'Combo', 'Resilience'])
                        modif_in = st.number_input('Rating:',
                                                   min_value=0,
                                                   max_value=5,
                                                   step=1,
                                                   key='type_modifier')
                        
                  with inputs[3]:
                        in_stats_type = st.selectbox('Wähle Type:',
                                     options=['Type'])
                        new_type = st.selectbox('Neuer Type:',
                                                options=df_elo['Type'].unique())
                        
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              update_stats(deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, save_cols, hist_cols, db)
                        
      else:
            st.title('Keine Berechtigung')
