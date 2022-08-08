#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 10:47:10 2022

@author: christoph
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from emoji import emojize
from scipy.optimize import curve_fit
from matplotlib import gridspec
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.patches as patches
from deta import Deta
from functions import *
import os
from dotenv import load_dotenv

load_dotenv(".env")
key = os.getenv('key')
deta = Deta(key)
db = deta.Base("ygo_elo_database")
st.set_page_config(page_title='YGO-Elo-Dashboard', page_icon=':trophy:' ,layout='wide')

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

# build data
df, hist_cols, save_cols = get_all_data(db)
df_elo = fetch_and_clean_data(df.copy())
hist_cols = sort_hist_cols(hist_cols)
df_select = df_elo.copy()
cols = df_elo.columns.to_list()
vgl_decks, all_in_one, vgl_player, vgl_style, ges_stats, update_spiele = st.tabs(['Vergleiche die Decks', 'Alles zu einem Deck','Vergleiche die Spieler', 'Vergleiche die Stile','Gesamtstatistik', 'Update neue Spiele'])

with vgl_decks:
      # filter
      left_column, mid_column, right_column = st.columns(3)
      with left_column:
            # filter for players
            owner = st.multiselect(
                  "Wähle Spieler:",
                  options = df_elo['Owner'].unique(),
                  default = df_elo['Owner'].unique())
            # filter for showing the data
            types_select = st.multiselect(
                  'Wähle Decktyp',
                  options = df_elo['Type'].unique(),
                  default = df_elo['Type'].unique())
            
      with mid_column:
            # filter for gruppe
            tier = st.multiselect(
                  "Wähle die Deckstärke:",
                  options = df_elo['Tier'].unique(),
                  default = df_elo['Tier'].unique()
                  )
            # filter for showing the data
            sorting = st.multiselect(
                  'Sortiere nach',
                  options = ['Elo-Rating', 'Elo-Verbesserung', 'mittlere Elo'],
                  default = ['Elo-Rating'])
            
            
      with right_column:
            # filter for modus
            tournament = st.multiselect(
                  'Suche nach einem Turnier:',
                  options = ['Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal'],
                  default = [])
            as_des = st.radio(
                  ' ',
                  options=['Absteigend', 'Aufsteigend'])
            
            
      slider_range = st.slider('Stellle den Elo-Bereich ein:', 
                min_value = int(0.95*df_elo['Elo'].min()),
                max_value = int(1.05*df_elo['Elo'].max()),
                value = [int(0.95*df_elo['Elo'].min()), int(1.05*df_elo['Elo'].max())],
                step = 1)
                                                       
      
      for j in range(len(sorting)):
            if sorting[j] == 'Elo-Rating':
                  sorting[j] = 'Elo'
            elif sorting[j] == 'Elo-Verbesserung':
                  sorting[j] = '1/4 Differenz'
            elif sorting[j] == 'mittlere Elo':
                  sorting[j] = 'Mean 1 Year'
                  
      for idx_tour, tour in enumerate(tournament):
            if idx_tour == 0:
                  df_select = df_elo[df_elo[tour]>0]
            else:
                  df_select = df_select[df_select[tour]>0]
      
      df_select = df_select.query(
            'Owner == @owner & Tier == @tier & Type == @types_select'
            )
      df_select = df_select[(df_select['Elo']>=slider_range[0]) & (df_select['Elo']<=slider_range[1])]
      df_select = df_select.reset_index()
      
      if len(df_select)>0:
            if as_des == 'Aufsteigend':
                  asdes = True
            else:
                  asdes = False
                  
            df_select = df_select.sort_values(by=sorting, ascending=asdes)
            
      
     
      st.subheader('Wähle deine Decks')
      deck = st.multiselect(
            '',
             options = df_select['Deck'].unique(),
             default = [])
      
      vergleiche_die_decks(df_elo, hist_cols, df_select, deck)
            
with all_in_one:
      deck_i = st.selectbox('Wähle ein deck', 
                                  options=df_elo['Deck'].unique())
                                  
      alles_zu_einem_deck(deck_i, df_elo, hist_cols)
                  
with vgl_player:
      vergleiche_die_spieler(df_elo, hist_cols)
          

with vgl_style:            
      vergleiche_stile(df_elo)
           
with ges_stats:
      Label = ['Kartenstapel','Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0']
      values = np.array(df_elo['Elo'].values).squeeze()
      
      counts,cbins = do_histogram(values, b=10, d=True)
      H, bins = do_histogram(values, b=10, d=False)
      
      mean = np.mean(values)
      std = np.std(values)
      param, opt = curve_fit(gauss, cbins, counts, bounds=[[1300, 0], [mean, std]])
      
      x = np.linspace(mean-4*std, mean+4*std, 500)
      a = 0.3

      fig = plt.figure()
      fig.set_figheight(10)
      fig.set_figwidth(15)
      
      # create grid for different subplots
      spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0, height_ratios=[6, 1])
      axs = fig.add_subplot(spec[0])
      axs1 = fig.add_subplot(spec[1])
      
      axs2 = axs.twinx()
      # 68% Intervall
      axs2.fill_between(np.linspace(mean-std, mean+std, 100), gauss(np.linspace(mean-std, mean+std, 100), mean, std), color='b', label='68% Interv.', alpha=a)
      # 95% Intervall
      axs2.fill_between(np.linspace(mean-2*std, mean-std, 100), gauss(np.linspace(mean-2*std, mean-std, 100), mean, std), color='g', label='95% Interv.', alpha=a)
      axs2.fill_between(np.linspace(mean+std, mean+2*std, 100), gauss(np.linspace(mean+std, mean+2*std, 100), mean, std), color='g', alpha=a)
      # Signifikanter Bereich
      axs2.fill_between(np.linspace(mean-4*std, mean-2*std, 100), gauss(np.linspace(mean-4*std, mean-2*std, 100), mean, std), color='r', label='Sign. Bereich', alpha=a)
      axs2.fill_between(np.linspace(mean+2*std, mean+4*std, 100), gauss(np.linspace(mean+2*std, mean+4*std, 100), mean, std), color='r', alpha=a)
      # Verteilungsfunktion
      axs2.plot(x, gauss(x, mean, std), label='PDF', color='gray')
      #
      axs2.plot([mean+2*std, mean+2*std], [0, gauss(mean+2*std, mean, std)], color='gray')
      axs2.text(mean+2*std, gauss(mean+2*std, mean, std),'%.f' % (mean+2*std), fontsize=15, color='white')

      axs2.plot([mean+std, mean+std], [0, gauss(mean+std, mean, std)], color='gray')
      axs2.text(mean+std, gauss(mean+std, mean, std),'%.f' % (mean+std), fontsize=15, color='white')

      axs2.plot([mean, mean], [0, gauss(mean, mean, std)], color='gray')
      axs2.text(mean, gauss(mean, mean, std), '%.f' % (mean), fontsize=15, color='white')
      
      axs2.plot([mean-std, mean-std], [0, gauss(mean-std, mean, std)], color='gray')
      axs2.text((mean-std)-20, gauss(mean-std, mean, std),'%.f' % (mean-std), fontsize=15, color='white')

      axs2.plot([mean-2*std, mean-2*std], [0, gauss(mean-2*std, mean, std)], color='gray')
      axs2.text((mean-2*std)-20, gauss(mean-2*std, mean, std),'%.f' % (mean-2*std), fontsize=15, color='white')

      # Data
      axs.bar(bins, H, abs(bins[1]-bins[0]), alpha=0.65, color='gray', label='Deck Histogram')

      # Layout
      axs.set_ylabel('Anzahl der Decks', fontsize=20, color='white')
      axs2.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=20, color='white')
      axs.grid()
      axs.legend(loc='upper left', fontsize=15)
      axs2.legend(loc='upper right', fontsize=15)
      axs2.set_ylim([0, 1.1*np.max(gauss(x, mean, std))])
      axs.set_xlim([mean-4*std, mean+4*std])
      
      # markiere die bereiche
      axs1.fill_between([mean-4*std, mean-2*std], [1, 1], color='r', alpha=a)
      axs1.fill_between([mean-2*std, mean-std], [1, 1], color='g', alpha=a)
      axs1.fill_between([mean-std, mean+std], [1, 1], color='b', alpha=a)
      axs1.fill_between([mean+std, mean+2*std], [1, 1], color='g', alpha=a)
      axs1.fill_between([mean+2*std, mean+4*std], [1, 1], color='r', alpha=a)
      # abgrenzung der bereiche
      axs1.plot([mean-2*std, mean-2*std], [0, 1], color='gray')
      axs1.plot([mean-std, mean-std], [0, 1], color='gray')
      axs1.plot([mean, mean], [0, 1], color='gray')
      axs1.plot([mean+std, mean+std], [0, 1], color='gray')
      axs1.plot([mean+2*std, mean+2*std], [0, 1], color='gray')
      # text in den bereichen
      offset1 = 25
      offset = 20
      
      axs1.xaxis.label.set_color('white')
      axs1.yaxis.label.set_color('white')
      axs2.xaxis.label.set_color('white')
      axs2.yaxis.label.set_color('white')
      
      axs1.spines['bottom'].set_color('white')
      axs1.spines['left'].set_color('white')
      axs1.spines['top'].set_color('white')
      axs1.spines['right'].set_color('white')
      axs1.tick_params(colors='white', which='both')
      
      axs2.spines['bottom'].set_color('white')
      axs2.spines['left'].set_color('white')
      axs2.spines['top'].set_color('white')
      axs2.spines['right'].set_color('white')
      axs2.tick_params(colors='white', which='both')
      
      axs1.text(mean-3*std-offset, 0.65, Label[0], fontsize=20, color='white')
      N = len(values[(values>=mean-4*std)&(values<mean-2*std)])
      axs1.text(mean-3*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
      
      axs1.text(mean-1.5*std-offset, 0.65, Label[1], fontsize=20, color='white')
      N = len(values[(values>=mean-2*std)&(values<mean-std)])
      axs1.text(mean-1.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
      
      axs1.text(mean-0.5*std-offset, 0.65, Label[2], fontsize=20, color='white')
      N = len(values[(values>=mean-std)&(values<mean)])
      axs1.text(mean-0.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
      
      axs1.text(mean+0.5*std-offset, 0.65, Label[3], fontsize=20, color='white')
      N = len(values[(values>=mean)&(values<mean+std)])
      axs1.text(mean+0.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
      
      axs1.text(mean+1.5*std-offset, 0.65, Label[4], fontsize=20, color='white')
      N = len(values[(values>=mean+std)&(values<mean+2*std)])
      axs1.text(mean+1.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
      
      axs1.text(mean+3*std-offset, 0.65, Label[5], fontsize=20, color='white')
      N = len(values[(values>=mean+2*std)&(values<mean+4*std)])
      axs1.text(mean+3*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
      
      axs1.set_xlim([mean-4*std, mean+4*std])
      axs1.set_xlabel('ELO', fontsize=20, color='white')
      axs1.set_ylim([0, 1])
      axs1.set(yticklabels=[])  # remove the tick labels
      axs1.tick_params(left=False)  # remove the ticks
      axs1.grid()
      axs1.tick_params(axis='x', labelsize=15, color='white')
      axs1.tick_params(axis='y', labelsize=15, color='white') 
      axs2.tick_params(axis='y', labelsize=15, color='white') 
      
      col0, col1 = st.columns(2)
      with col0:
            st.pyplot(fig, transparent=True)
            
      with col1:
            n = len(df_elo['Deck'].unique())
            df_max = df_elo[hist_cols+['Elo']].agg(['idxmax', 'max'])
            df_min = df_elo[df_elo[hist_cols+['Elo']]>0][hist_cols+['Elo']].agg(['idxmin', 'min'])
            st.subheader(f"     Decks in Wertung {n}")
            st.subheader(f"     Matches: {int(df_elo['Matches'].sum())/2}")
            st.subheader(f"     Spiele: {int(df_elo['Siege'].sum()+df_elo['Remis'].sum()+df_elo['Niederlage'].sum())/2}")
            st.subheader('')
            st.subheader(f"     Höchste Elo:             {int(df_max.loc['max', :].max())}, {df_elo.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']}")
            st.subheader(f"     Aktuelle beste Elo:      {int(df_max.at['max', 'Elo'])}, {df_elo.at[int(df_max.at['idxmax', 'Elo']), 'Deck']}")
            st.subheader('')
            st.subheader(f"     Niedrigste Elo:          {int(df_min.loc['min', :].min())}, {df_elo.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']}")
            st.subheader(f"     Aktuelle niedrigste Elo: {int(df_min.loc['min', 'Elo'].min())}, {df_elo.at[df_min.loc['idxmin', 'Elo'], 'Deck']}")
            
            
            n_decks = len(df_elo['Deck'].unique())

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