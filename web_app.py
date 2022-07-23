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

def make_spider_plot(df):
      categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
      
      angles = [n / float(6) * 2 * np.pi for n in range(6)]
      angles += angles[:1]
      angles = (np.array(angles) + np.pi/2)%(2*np.pi)
      # Initialise the spider plot
      fig, axs = plt.subplots(figsize=(1,1), subplot_kw=dict(polar=True))
      # Draw one axe per variable + add labels
      plt.xticks(angles[:-1], categories, color='white', size=7)
      # Draw ylabels
      #plt.rlabel_position(0)
      axs.set_yticks([1,2,3, 4, 5])
      axs.set_yticklabels(['', '', '', '', ''], color="white", size=3)
      axs.set_ylim([0,5.3])
 
      #Plot data
      axs.plot(angles, np.append(df, df[0]), linewidth=1, linestyle='solid', color='orange')
 
      # Fill area
      axs.fill(angles, np.append(df, df[0]), 'orange', alpha=0.3)
      return fig

def semi_circle_plot(val):
      val = np.append(val, sum(val))  # 50% blank
      colors = ['green', 'blue', 'red', "#00172B"]
      explode= 0.05*np.ones(len(val))
      # plot
      fig = plt.figure(figsize=(8,8))
      ax = fig.add_subplot(1,1,1)
      #p = patches.Rectangle((left, bottom), width, height,fill=False, transform=ax.transAxes, clip_on=False)

      ax.pie(val, colors=colors, pctdistance=0.85, explode=explode)
      ax.text(-1.05, 1.1, f"N {int(val[2]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
      ax.text(-0.1, 1.1, f"U {int(val[1]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
      ax.text(0.65, 1.1, f"S {int(val[0]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
      ax.add_artist(plt.Circle((0, 0), 0.6, color="#00172B"))
      
      return fig

st.set_page_config(page_title='YGO-Elo-Dashboard', page_icon=':trophy:' ,layout='wide')

# --- USER AUTHENTICATION ---
with st.sidebar:

      names = ['Chriss', 'Rebecca Briggs']
      usernames = ['Chriss', 'rbriggs']
      
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

def fetch_and_clean_data():
       df_elo = pd.read_csv('./web_app_table.csv')
       # process data
       cols = df_elo.columns.to_list()
       df_elo['Elo'] = df_elo['Elo'].astype(int)
       df_elo['1/4 Differenz'] = df_elo['Elo']-df_elo[cols[-1]]
       df_elo['1/4 Differenz'] = df_elo['1/4 Differenz'].fillna(0).astype(int)
       df_elo['1/2 Differenz'] = df_elo['Elo']-df_elo[cols[-2]]
       df_elo['1/2 Differenz'] = df_elo['1/2 Differenz'].fillna(0).astype(int)
       df_elo['1 Differenz'] = df_elo['Elo']-df_elo[cols[-4]]
       df_elo['1 Differenz'] = df_elo['1 Differenz'].fillna(0).astype(int)
       df_elo['Mean 1 Year'] = np.mean(df_elo[cols[-5:-1]+['Elo']],1)
       df_elo['Mean 1 Year'] = df_elo['Mean 1 Year'].fillna(0).astype(int)
       df_elo['Std. 1 Year'] = np.std(df_elo[cols[-5:-1]+['Elo']],1)
       df_elo['Std. 1 Year'] = df_elo['Std. 1 Year'].fillna(0).astype(int)
       
       return df_elo.fillna(0)
 
# build data
df_elo = fetch_and_clean_data()
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
      plot, stats = st.tabs(['Elo-Verlauf', 'Statistiken'])
      with plot:
            with st.container():
                  # plot
                  fig, axs = plt.subplots(1, figsize=(10,4))
                  axs.spines['bottom'].set_color('white')
                  axs.spines['left'].set_color('white')
                  axs.tick_params(colors='white', which='both')
                  axs.set_xlabel('Month/Year', fontsize=14)
                  axs.set_ylabel('Elo-Rating', fontsize=14)
                  axs.xaxis.label.set_color('white')
                  axs.yaxis.label.set_color('white')
                  tmp = [0]
                  for i in range(len(deck)):
                        idx = int(df_select[df_select['Deck']==deck[i]].index.to_numpy())
                        tmp = np.append(tmp, df_select.loc[idx, cols[20:-5]+['Elo']].to_numpy())
                        axs.plot(df_select.loc[idx, cols[20:-5]+['Elo']], label=df_select.at[idx, 'Deck'], lw=3)
                        
                        tmp = np.array(tmp)
                        tmp_min = tmp[tmp>0]
                        axs.set_ylim([0.95*np.min(tmp_min), 1.05*np.max(tmp)])
                  axs.legend(loc='lower left')
                  axs.grid()
                   
                  
                  st.subheader('Plot')
                  st.pyplot(fig, transparent=True) 
      with stats:   
            st.markdown('# Überblick')
            if len(deck) > 0:
                  n_rows = int(np.ceil(len(deck)/4))
                  counter = 0
                  for r in range(n_rows):
                        columns = st.columns([2,3,2,3,2,3,2,3])
                        for i in range(4):
                              if counter >= len(deck):
                                    break
                              
                              idx = int(df_select[df_select['Deck']==deck[counter]].index.to_numpy())
                              values = df_select.loc[idx, ['Siege', 'Remis', 'Niederlage']].to_list()
                              
                              with columns[2*i]:
                                    st.header(df_select.at[idx, 'Deck']+" :star:"*int(df_select.at[idx, 'Wanderpokal']))
                                    st.subheader(df_select.at[idx, 'Tier'])
                                    st.subheader(f"Matches: {int(df_select.at[idx, 'Matches'])}")
                                    st.subheader(f"Spiele: {int(np.sum(values))}")
                                    st.metric('Aktuelle Elo', df_select.at[idx, 'Elo'], int(df_select.at[idx, '1/4 Differenz']))
                                    st.metric('Gegner Stärke', int(df_select.at[idx, 'dgp']))
                                    st.caption('Wanderpokal:   ' + ':star:'*int(df_select.at[idx, 'Wanderpokal']))
                                    st.caption('Fun Pokal:     ' + ':star:'*int(df_select.at[idx, 'Fun Pokal']))
                                    st.caption('Meisterschaft: ' + ':star:'*int(df_select.at[idx, 'Meisterschaft']))
                                    st.caption('Liga Pokal:    ' + ':star:'*int(df_select.at[idx, 'Liga Pokal']))
                                    
                              with columns[2*i+1]:
                                    st.header(" ")
                                    st.header(" ")
                                    
                                    fig = semi_circle_plot(values)
                                    st.pyplot(fig, transparent=True)
                                    
                                    categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
                                    fig = make_spider_plot(df_select.loc[idx, categories].astype(int).to_numpy())
                                    st.pyplot(fig, transparent=True)
                                    
                                    counter += 1
      
with all_in_one:
      deck_i = st.selectbox('Wähle ein deck', 
                            options=df_elo['Deck'].unique())
      if len(deck_i) > 0:
            idx_deck = int(df_elo[df_elo['Deck']==deck_i].index.to_numpy())
            deck_cols = st.columns([3,3,3,4,5])
            tmp_df = df_elo.copy()
            tmp_df = tmp_df[['Deck', 'Elo']].sort_values(by=['Elo'], ascending=False)
            platz = int(tmp_df[tmp_df['Deck']==deck_i].index.to_numpy())+1
            percentage = int(platz/len(df_elo)*1000)/10
            with deck_cols[0]:
                  st.subheader(df_elo.at[idx_deck, 'Tier'])
                  st.subheader(df_elo.at[idx_deck, 'Type'])
                  st.subheader('')
                  st.subheader('Elo-Meilensteine:')
                  st.caption(f"Beste Elo: {int(df_elo.loc[idx_deck, cols[20:-5]].max())}")
                  st.caption(f"Schlechteste Elo: {int(df_elo.loc[idx_deck, cols[20:-5]].min())}")
                  st.subheader("")
                  st.subheader("Turnier")
                  st.caption('Wanderpokal:   ' + ':star:'*int(df_elo.at[idx_deck, 'Wanderpokal']))
                  st.caption('Fun Pokal:     ' + ':star:'*int(df_elo.at[idx_deck, 'Fun Pokal']))
                  st.caption('Meisterschaft: ' + ':star:'*int(df_elo.at[idx_deck, 'Meisterschaft']))
                  st.caption('Liga Pokal:    ' + ':star:'*int(df_elo.at[idx_deck, 'Liga Pokal']))
                  
            with deck_cols[1]:
                  st.metric('Aktuelle Elo', df_elo.at[idx_deck, 'Elo'], int(df_elo.at[idx_deck, '1/4 Differenz']))
                  st.subheader('Spiele & Matches')
                  st.caption(f"Matches: {int(df_elo.at[idx_deck, 'Matches'])}")
                  st.caption(f"Spiele: {int(df_elo.at[idx_deck, 'Siege']+df_elo.at[idx_deck, 'Remis']+df_elo.at[idx_deck, 'Niederlage'])}")
                  
                  
            with deck_cols[2]:
                  st.metric('Gegner Stärke', int(df_elo.at[idx_deck, 'dgp']))
                  st.subheader('Platzierung:')
                  st.caption(f"Aktueller Rang: {platz}")
                  st.caption(f"Obere {percentage}% aller Decks")
            
                  
            with deck_cols[3]:
                  fig, axs = plt.subplots(1,1, figsize=(5, 3))
                  axs.plot(range(5), df_elo.loc[idx_deck, cols[-9:-5]+['Elo']].to_numpy())
                  axs.set_title('Eloverlauf des letzen Jahres', color='white')
                  axs.set_xticks(range(5))
                  axs.set_xticklabels(cols[-9:-5]+['Elo'], color='white', rotation=45)
                  axs.set_ylabel('Elo', color="white")
                  axs.grid()
                  axs.spines['bottom'].set_color('white')
                  axs.spines['left'].set_color('white')
                  axs.tick_params(colors='white', which='both')
                  st.pyplot(fig, transparent=True)
                  
                  fig = semi_circle_plot(df_elo.loc[idx_deck, ['Siege', 'Remis', 'Niederlage']].to_list())
                  st.text('Spielergebnisse')
                  st.pyplot(fig, transparent=True)
                  categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
                  
            with deck_cols[4]:
                  fig = make_spider_plot(df_elo.loc[idx_deck, categories].astype(int).to_numpy())
                  st.pyplot(fig, transparent=True)
                  
def get_plyer_infos(df, name, cols):
      cols_spider = ['Attack', 'Control', 'Recovery', 'Consistensy','Combo', 'Resilience']
      n_wp = df[df['Owner']==name]['Wanderpokal'].sum()
      n_fun = df[df['Owner']==name]['Fun Pokal'].sum()
      act_elo_best = df[df['Owner']==name]['Elo'].astype(int).max()
      act_elo_min = df[df['Owner']==name]['Elo'].astype(int).min()
      
      tmp = df[df['Owner']==name][cols[18:-5]].astype(int).max()
      hist_elo_best = int(np.max(tmp[tmp>0]))
      
      tmp = df[df['Owner']==name][cols[18:-5]].astype(int).min()
      hist_elo_min = int(np.min(tmp[tmp>0]))
      
      n_decks = len(df[df['Owner']==name])
      fig = make_spider_plot(np.round(df_elo[df_elo['Owner']==name][cols_spider].mean(), 0).astype(int).values)
      fig2 = make_deck_histo(df[df['Owner']==name])
      fig3 = make_type_histo(df, name)
      return n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min

def make_deck_histo(df):
      n1 = len(df[df['Tier']=='Kartenstapel'])
      n2 = len(df[df['Tier']=='Fun'])
      n3 = len(df[df['Tier']=='Good'])
      n4 = len(df[df['Tier']=='Tier 2'])
      n5 = len(df[df['Tier']=='Tier 1'])
      n6 = len(df[df['Tier']=='Tier 0'])
      
      fig, axs = plt.subplots(1, figsize=(9,9))
      axs.bar([1,2,3,4,5,6], [n1, n2, n3, n4, n5, n6], color='gray')
      axs.set_xticks([1,2,3,4,5,6])
      axs.set_xticklabels(['Kartenstapel', 'Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0'], rotation=-45, color='white')
      axs.set_ylabel('Anzahl Decks', color='white')
      axs.grid()
      axs.set_title('Verteilung der Decks auf die Spielstärken', color='white')
      axs.spines['bottom'].set_color('white')
      axs.spines['left'].set_color('white')
      axs.tick_params(colors='white', which='both')
      return fig
      
def make_type_histo(df, name):
      types = df['Type'].unique()
      n_types = len(types)
      fig, axs = plt.subplots(1, figsize=(8,8))
      n_decks = np.zeros(n_types)
      for k in range(n_types):
            n_decks[k] = len(df[(df['Type']==types[k])&(df['Owner']==name)])
      axs.bar(range(n_types), n_decks, color='gray')
      axs.set_xticks(range(n_types))
      axs.set_xticklabels(types, rotation=-45, color='white')
      axs.set_ylabel('Anzahl Decks', color='white')
      axs.grid()
      axs.set_title('Verteilung der Decks auf die Typen', color='white')
      axs.spines['bottom'].set_color('white')
      axs.spines['left'].set_color('white')
      axs.tick_params(colors='white', which='both')
      return fig

with vgl_player:
      player_cols = st.columns(5)
      # Chriss
      with player_cols[0]:
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Christoph', cols)
            st.header('Chirstoph ')
            st.text(f'Decks in Wertung: {n_decks}')
            df = df_elo[df_elo['Owner']=='Christoph'][['Siege', 'Remis', 'Niederlage']].sum()
            fig = semi_circle_plot(df.values)
            st.text(f"Matches: {int(df_elo[df_elo['Owner']=='Christoph']['Matches'].sum()//2)}")
            st.text(f"Spiele: {int(df.sum()//2)}")
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig1, transparent=True)
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)
            
      # Finn
      with player_cols[1]:
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Finn', cols)
            st.header('Finn')
            st.text(f'Decks in Wertung: {n_decks}')
            st.text(f"Matches: {int(df_elo[df_elo['Owner']=='Finn']['Matches'].sum()//2)}")
            df = df_elo[df_elo['Owner']=='Finn'][['Siege', 'Remis', 'Niederlage']].sum()
            fig = semi_circle_plot(df.values)
            st.text(f"Spiele: {int(df.sum()//2)}")
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig1, transparent=True)
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)

            
      # Frido
      with player_cols[2]:
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Frido', cols)
            st.header('Frido')
            st.text(f'Decks in Wertung: {n_decks}')
            df = df_elo[df_elo['Owner']=='Frido'][['Siege', 'Remis', 'Niederlage']].sum()
            fig = semi_circle_plot(df.values)
            st.text(f"Matches: {int(df_elo[df_elo['Owner']=='Frido']['Matches'].sum()//2)}")
            st.text(f"Spiele: {int(df.sum()//2)}")
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig1, transparent=True)
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)

      # Jan
      with player_cols[3]:
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Jan', cols)
            st.header('Jan')
            st.text(f'Decks in Wertung: {n_decks}')
            df = df_elo[df_elo['Owner']=='Jan'][['Siege', 'Remis', 'Niederlage']].sum()
            fig = semi_circle_plot(df.values)
            st.text(f"Matches: {int(df_elo[df_elo['Owner']=='Jan']['Matches'].sum()//2)}")
            st.text(f"Spiele: {int(df.sum()//2)}")
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig1, transparent=True)
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)
            
            
      # Thomas
      with player_cols[4]:
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Thomas', cols)
            st.header('Thomas')
            st.text(f'Decks in Wertung: {n_decks}')
            df = df_elo[df_elo['Owner']=='Thomas'][['Siege', 'Remis', 'Niederlage']].sum()
            fig = semi_circle_plot(df.values)
            st.text(f"Matches: {int(df_elo[df_elo['Owner']=='Thomas']['Matches'].sum()//2)}")
            st.text(f"Spiele: {int(df.sum()//2)}")
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig1, transparent=True)
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)
            
            
with vgl_style:
      histo, viobox = st.columns(2)
      types = df_elo['Type'].unique()
      with histo:
            n_types = len(types)
            fig, axs = plt.subplots(1, figsize=(8,8))
            n_decks = np.zeros(n_types)
            for k in range(n_types):
                  n_decks[k] = len(df_elo[df_elo['Type']==types[k]])
            axs.bar(range(n_types), n_decks, color='gray')
            axs.set_xticks(range(n_types))
            axs.set_xticklabels(types, rotation=-45, color='white')
            axs.set_ylabel('Anzahl Decks', color='white')
            axs.grid()
            axs.set_title('Verteilung der Decks auf die Typen', color='white')
            axs.spines['bottom'].set_color('white')
            axs.spines['left'].set_color('white')
            axs.tick_params(colors='white', which='both')
            st.pyplot(fig, transparent=True)
            
      with viobox:
            fig, axs = plt.subplots(1, figsize=(8,8))
            for idx, tmp_type in enumerate(types):
                  axs.violinplot(df_elo[df_elo['Type']==tmp_type]['Elo'], positions=[idx], showmeans=True, showextrema=False, showmedians=False)
                  axs.boxplot(df_elo[df_elo['Type']==tmp_type]['Elo'], positions=[idx])
            axs.set_title('Eloverteilung pro Decktyp', color='white')
            axs.set_ylabel('Elo-Rating', color='white')
            axs.set_xticks(range(n_types))
            axs.set_xticklabels(types, rotation=-45, color='white')
            axs.grid()
            
            axs.spines['bottom'].set_color('white')
            axs.spines['left'].set_color('white')
            axs.tick_params(colors='white', which='both')
            st.pyplot(fig, transparent=True)

def gauss(x, m, s):
      return 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2/(4*s**2))

# Histogramfunktion
def do_histogram(data, b=10, d=True):
    counts, bins = np.histogram(data, b, density=d)

    n = np.size(bins)
    cbins = np.zeros(n-1)

    for ii in range(n-1):
        cbins[ii] = (bins[ii+1]+bins[ii])/2

    return counts, cbins

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
            df_max = df_elo[cols[19:-5]].agg(['idxmax', 'max'])
            df_min = df_elo[df_elo[cols[19:-5]]>0][cols[19:-5]].agg(['idxmin', 'min'])
            st.subheader(f"     Decks in Wertung {n}")
            st.subheader(f"     Matches: {int(df_elo['Matches'].sum())/2}")
            st.subheader(f"     Spiele: {int(df_elo['Siege'].sum()+df_elo['Remis'].sum()+df_elo['Niederlage'].sum())/2}")
            st.subheader('')
            st.subheader(f"     Höchste Elo:             {int(df_max.loc['max', :].max())}, {df_elo.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']}")
            st.subheader(f"     Aktuelle beste Elo:      {int(df_max.at['max', cols[-7]])}, {df_elo.at[int(df_max.at['idxmax', cols[-7]]), 'Deck']}")
            st.subheader('')
            st.subheader(f"     Niedrigste Elo:          {int(df_min.loc['min', :].min())}, {df_elo.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']}")
            st.subheader(f"     Aktuelle niedrigste Elo: {int(df_min.loc['min', cols[-7]].min())}, {df_elo.at[df_min.loc['idxmin', cols[-7]], 'Deck']}")
            
            
            n_decks = len(df_elo['Deck'].unique())
    
def update_tiers(df):
      Label = ['Kartenstapel','Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0']
      NCl = len(Label)
      
      mean = int(df['Elo'].mean())
      std = float(df['Elo'].std())
      bins = np.array([0, mean-2*std, mean-std, mean, mean+std, mean+2*std, 10e6])
      liste =[]
      points = []
      
      for jj in range(len(df)):    
          for kk in range(NCl):
              if(df.at[jj, 'Elo']>=bins[kk] and df.at[jj, 'Elo']<=bins[kk+1]):
                  liste.append(Label[kk])
                  break
      
      df['Tier'] = liste
      return df
      
def update_elo_ratings(deck1, deck2, erg1, erg2, df_elo):
      # get index
      idx1 = int(df_elo[df_elo['Deck']==deck1].index.to_numpy())
      idx2 = int(df_elo[df_elo['Deck']==deck2].index.to_numpy())
      # update siege/niederlagen
      if not erg1 == erg2:
            df_elo.at[idx1, 'Siege'] += erg1
            df_elo.at[idx1, 'Niederlage'] += erg2
            
            df_elo.at[idx2, 'Siege'] += erg2
            df_elo.at[idx2, 'Niederlage'] += erg1
      else:
            df_elo.at[idx1, 'Remis'] +=erg1
            df_elo.at[idx2, 'Remis'] +=erg2
      # update matches
      df_elo.at[idx1, 'Matches'] += 1
      df_elo.at[idx2, 'Matches'] += 1
      # update dgp
      df_elo.at[idx1, 'dgp'] = (2*df_elo.at[idx1, 'dgp'] + df_elo.at[idx2, 'Elo'])//3
      df_elo.at[idx2, 'dgp'] = (2*df_elo.at[idx2, 'dgp'] + df_elo.at[idx1, 'Elo'])//3
      # update elo
      norm = erg1 + erg2 
      score1 = erg1/norm
      score2 = erg2/norm
      
      # calculate winning probabilities 
      # Formula from Chess ELO System
      alpha = 10**((df_elo.at[idx1, 'Elo']-df_elo.at[idx2, 'Elo'])/400)
      p1 = alpha/(alpha+1)
      p2 = 1-p1
      
      # Update ELO (Chess Formula)
      df_elo.at[idx1, 'Elo'] += int(32*(score1-p1))
      df_elo.at[idx2, 'Elo'] += int(32*(score2-p2))
      
      df_elo = update_tiers(df_elo)
      df_elo[cols[:-5]].to_csv('./web_app_table.csv', index=False)
            

def update_stats(deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, cols):
      idx = int(df_elo[df_elo['Deck']==deck_choose].index.to_numpy())
      if len(in_stats) > 0:
            df_elo.at[idx, in_stats] = modif_in

      if len(in_stats_type)>0:
            df_elo.at[idx, in_stats_type] = new_type
            
      df_elo[cols[:-5]].to_csv('./web_app_table.csv', index=False)
      
def update_tournament(deck, tour, df, cols):
      idx = int(df[df['Deck']==deck].index.to_numpy())
      df.at[idx, tour] += 1
      df[cols[:-5]].to_csv('./web_app_table.csv', index=False)
      
def update_history(name, df, cols):
      if not name in cols:
            df[name] = df['Elo']
      
            df[cols[:-5]+[name]].to_csv('./web_app_table.csv', index=False)
      
def insert_new_deck(new_deck, owner, attack, control, resilience, recovery, combo, deck_type, df, cols):
      n = len(df)
      df.at[n, 'Deck'] = new_deck
      df.at[n, 'Attack'] = attack
      df.at[n, 'Control'] = control
      df.at[n, 'Resilience'] = resilience
      df.at[n, 'Recovery'] = recovery
      df.at[n, 'Combo'] = combo
      df.at[n, 'Type'] = deck_type
      df.at[n, 'Owner'] = owner
      df.at[n, 'Elo'] = 1500
      df.at[n, 'Tier'] = 'Tier 2'
      df = df.fillna(0)
      df[cols[:-5]].to_csv('./web_app_table.csv', index=False)
      
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
                              update_elo_ratings(deck1, deck2, erg1, erg2, df_elo.copy())
                   
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
                              update_tournament(deck_tour, tournament, df_elo.copy(), cols)
                  
            
            st.header('Trage neues Deck ein:')
            with st.form('neies_deck'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        new_deck = st.text_input('Names des neuen Decks', key='new deck')
                        owner = st.text_input('Names des Spielers', key='player')
                  with inputs[2]:
                        attack = st.number_input('Attack-Rating', min_value=0, max_value=5, step=1, key='attack')
                        control = st.number_input('Control-Rating', min_value=0, max_value=5, step=1, key='control')
                        resilience = st.number_input('Resilience-Rating', min_value=0, max_value=5, step=1, key='resilience')
                        
                  with inputs[3]:
                        recovery = st.number_input('Recovery-Rating', min_value=0, max_value=5, step=1, key='recovery')
                        combo = st.number_input('Combo-Rating', min_value=0, max_value=5, step=1, key='combo')
                        deck_type = st.selectbox('Wähle einen Decktype:',
                                                 options=df_elo['Type'].unique(), key='decktype')
                        
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              insert_new_deck(new_deck, owner, attack, control, resilience, recovery, combo, deck_type, df_elo.copy(), cols)
                              
            st.header('Update History:')
            with st.form('history_update'):
                  inputs = st.columns(6)
                  with inputs[1]:
                        date = st.text_input('Monta/Jahr des neuen Updates:', key='date')
                  with inputs[4]:
                        if st.form_submit_button("Submit"):
                              update_history(date, df_elo.copy(), cols)
                              
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
                              update_stats(deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, cols)
                        
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