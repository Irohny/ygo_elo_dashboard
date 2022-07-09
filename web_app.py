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

st.set_page_config(page_title='YGO Elo Dashboard', layout='wide')
# ---- HIDE STREAMLIT STYLE ----
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

#@st.cache
@st.cache(allow_output_mutation=True)
def fetch_and_clean_data():
       df_elo = pd.read_csv('./web_app_table.csv')
       return df_elo
 
df_elo = fetch_and_clean_data()
df_select = df_elo
cols = df_elo.columns
df_select[['Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal']] = df_elo[['Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal']].fillna(0).astype(int) 
df_select['Elo'] = df_elo[cols[-1]].astype(int)
df_select['1/4 Differenz'] = df_select['Elo']-df_elo[cols[-2]]
df_select['1/4 Differenz'] = df_select['1/4 Differenz'].fillna(0).astype(int)
df_select['1/2 Differenz'] = df_select['Elo']-df_elo[cols[-3]]
df_select['1/2 Differenz'] = df_select['1/2 Differenz'].fillna(0).astype(int)
df_select['1 Differenz'] = df_select['Elo']-df_elo[cols[-5]]
df_select['1 Differenz'] = df_select['1 Differenz'].fillna(0).astype(int)
df_select['Mean 1 Year'] = np.mean(df_elo[cols[-5:-1]],1)
df_select['Mean 1 Year'] = df_select['Mean 1 Year'].fillna(0).astype(int)
df_select['Std. 1 Year'] = np.std(df_elo[cols[-5:-1]],1)
df_select['Std. 1 Year'] = df_select['Std. 1 Year'].fillna(0).astype(int)



left_column, mid_column, right_column = st.columns(3)
with left_column:
      st.subheader('Filter for your deck:')
      # filter for players
      owner = st.multiselect(
            "Select Player:",
            options = df_elo['Owner'].unique(),
            default = df_elo['Owner'].unique())
      
      # filter for gruppe
      tier = st.multiselect(
            "Select Tier:",
            options = df_elo['Tier'].unique(),
            default = ['Tier 0', 'Tier 1', 'Tier 2']
            )
      
      # filter for modus
      tournament = st.multiselect(
            'Select your Tournament:',
            options = ['Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal'],
            default = [])
      
      # filter for showing the data
      sorting = st.multiselect(
            'Filter options:',
            options = ['Elo', '1/4 Differenz', '1/2 Differenz', '1 Differenz', 'Mean 1 Year'],
            default = ['Elo'])
      as_des = st.radio(
            'Ascending',
            options=['False', 'True'])
for idx_tour, tour in enumerate(tournament):
      if idx_tour == 0:
            df_select = df_elo[df_elo[tour]>0]
      else:
            df_select = df_select[df_select[tour]>0]

df_select = df_select.query(
      'Owner == @owner & Tier == @tier'
      )
df_select = df_select.reset_index()

if len(df_select)>0:
      if as_des == 'True':
            asdes = True
      else:
            asdes = False
            
      df_select = df_select.sort_values(by=sorting, ascending=asdes)
      

# mid layer
with mid_column:
      st.subheader('Plot your Decks')
      deck = st.multiselect(
            'Decks:',
            options = df_select['Deck'].unique(),
            default = [])

# plot
fig, axs = plt.subplots(1, figsize=(8,8))
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
      tmp = np.append(tmp, df_select.loc[idx, cols[7:]].to_numpy())
      axs.plot(df_select.loc[idx, cols[7:]], label=df_select.at[idx, 'Deck'], lw=3)
      
      tmp = np.array(tmp)
      tmp_min = tmp[tmp>0]
      axs.set_ylim([0.95*np.min(tmp_min), 1.05*np.max(tmp)])
axs.legend(loc='lower left')
axs.grid()
 
with right_column:
      st.subheader('Plot')
      st.pyplot(fig, transparent=True) 
st.markdown('# Tabelle')
display_cols = ['Deck', 'Owner', 'Tier', 'Elo','1/4 Differenz', '1/2 Differenz',
                '1 Differenz', 'Mean 1 Year','Std. 1 Year', 'Wanderpokal', 
                'Meisterschaft', 'Liga Pokal', 'Fun Pokal']
st.table(df_select[display_cols])
st.markdown('---')