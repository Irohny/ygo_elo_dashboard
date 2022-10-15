#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:41:31 2022

@author: christoph
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datetime
from emoji import emojize
from st_aggrid import AgGrid
from matplotlib import gridspec
import time

def get_all_data(db):
      # get all data
      res = db.fetch()
      # create a dataframe
      df = pd.DataFrame(res.items)
      # build a dataframe from the history
      tmp = df.loc[:, 'History'].to_list()
      hist_df = pd.DataFrame(tmp).fillna(0).astype(int)
      hist_cols = list(np.flip(hist_df.columns.to_list()))
      del df['History']
      save_cols = df.columns.to_numpy()
      save_cols = np.delete(save_cols, np.where(save_cols == 'key'))
      # merge history dataframe and rest
      df_all = pd.merge(df, hist_df, right_index=True, left_index=True)
      
      return df_all, hist_cols, save_cols

def get_deck_by_dict(db, con):
      res = db.fetch(con)
      df = pd.DataFrame(res.items)
      # build a dataframe from the history
      tmp = df.loc[:, 'History'].to_list()
      hist_df = pd.DataFrame(tmp).fillna(0).astype(int)
      hist_cols = list(np.flip(hist_df.columns.to_list()))
      del df['History']
      save_cols = df.columns.to_numpy()
      save_cols = np.delete(save_cols, np.where(save_cols == 'key'))
      # merge history dataframe and rest
      df_all = pd.merge(df, hist_df, right_index=True, left_index=True)
      #cleaning data
      df_all = fetch_and_clean_data(df_all)
      hist_cols = sort_hist_cols(hist_cols)
      return df_all, hist_cols, save_cols
      
def update_element(unique_key, ds, save_cols, hist_cols, db):
      '''
      Method for updating an element in the database
      '''
      dict_update = {}
      dict_hist = {}
      
      for col in save_cols:
            dict_update[col] = str(ds[col])
      for col in hist_cols:
            dict_hist[col] = str(ds[col])
      dict_update['History'] = dict_hist
      db.put(dict_update, unique_key)
      time.sleep(5)
      
def update_tournament(deck, tour, df, save_cols, hist_cols, db):
      idx = int(df[df['Deck']==deck].index.to_numpy())
      df.at[idx, tour] += 1
      unique_key = df.at[idx, 'key']
      update_element(unique_key, df.loc[idx, :], save_cols, hist_cols, db)
      
def update_stats(deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, save_cols, hist_cols, db):
      idx = int(df_elo[df_elo['Deck']==deck_choose].index.to_numpy())
      if len(in_stats) > 0:
            df_elo.at[idx, in_stats] = modif_in

      if len(in_stats_type)>0:
            df_elo.at[idx, in_stats_type] = new_type
      unique_key = df_elo.at[idx, 'key']
      update_element(unique_key, df_elo.loc[idx, :], save_cols, hist_cols, db)

def update_history(df, hist_cols, save_cols, db):
      date = datetime.date.today()
      name = date.strftime("%m/%Y")
      if not name in hist_cols:
            df[name] = df['Elo']
            hist_cols += [name]      
            for i in range(len(df)):
                  unique_key = df.at[i, 'key']
                  update_element(unique_key, df.loc[i, :], save_cols, hist_cols, db)
                  
def insert_new_deck(deck, name, atk, contr, rec, cons, combo, resi, typ, db):
      '''
      Function for inserting a new deck to the database
      '''
      input_dict = {
       'Deck':deck,	
       'Owner':name,
       'Tier':'Tier 2',	
       'Attack':atk,
       'Control':contr,
       'Recovery':rec,
       'Consistensy':cons, 
       'Combo':combo,	
       'Resilience':resi,
       'Type':typ,	
       'Wanderpokal':0,
       'Meisterschaft':0,
       'Liga Pokal':0,	
       'Fun Pokal':0, 	
       'Matches':0,	
       'Siege':0,	
       'Remis':0,	
       'Niederlage':0,	
       'dgp':0, 
       'Elo':1500,
       'History':{}}
      db.put(input_dict)
     
def fetch_and_clean_data(df, hist_cols):
      cols_to_int = ['Elo', 'dgp', 'Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal', 'Attack', 'Combo', 'Resilience', 'Recovery',
                        'Consistensy', 'Control','Matches', 'Siege', 'Remis', 'Niederlage']
      for k in cols_to_int:
            df[k] = df[k].astype(float).fillna(0)
            df[k] = df[k].astype(int)

      df['1/4 Differenz'] = df['Elo']-df[hist_cols[-1]]
      df['1/4 Differenz'] = df['1/4 Differenz'].fillna(0).astype(int)
      df['1/2 Differenz'] = df['Elo']-df[hist_cols[2]]
      df['1/2 Differenz'] = df['1/2 Differenz'].fillna(0).astype(int)
      df['1 Differenz'] = df['Elo']-df[hist_cols[-4]]
      df['1 Differenz'] = df['1 Differenz'].fillna(0).astype(int)
      df['Mean 1 Year'] = np.mean(df[hist_cols[-5:-1]+['Elo']],1)
      df['Mean 1 Year'] = df['Mean 1 Year'].fillna(0).astype(int)
      df['Std. 1 Year'] = np.std(df[hist_cols[-5:-1]+['Elo']],1)
      df['Std. 1 Year'] = df['Std. 1 Year'].fillna(0).astype(int)
       
      return df.fillna(0)

def sort_hist_cols(hist_cols):
      t = []
      for c in hist_cols:
          t.append(datetime.date(year = int(c[-4:]), month = int(c[:2]), day=1))
      t = pd.Series(t).sort_values().reset_index(drop=True)
      out = []
      for i in range(len(t)):
            out.append(t[i].strftime('%m/%Y')) 
      return out

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_elo_history(df_select, deck, hist_cols):
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
            tmp = np.append(tmp, df_select.loc[idx, hist_cols+['Elo']].to_numpy())
            axs.plot(df_select.loc[idx, hist_cols+['Elo']], label=df_select.at[idx, 'Deck'], lw=3)
            
            tmp = np.array(tmp)
            tmp_min = tmp[tmp>0]
            axs.set_ylim([0.95*np.min(tmp_min), 1.05*np.max(tmp)])
            
      axs.set_xticks(range(len(hist_cols)+1))
      axs.set_xticklabels(hist_cols+['Elo'], rotation=45)
      axs.legend(loc='lower left')
      axs.grid()
      return fig

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
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

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def semi_circle_plot(val):
      val = np.append(val, sum(val))  # 50% blank
      colors = ['green', 'blue', 'red', "#00172B"]
      explode= 0.05*np.ones(len(val))
      # plot
      fig = plt.figure(figsize=(8,5))
      ax = fig.add_subplot(1,1,1)
      #p = patches.Rectangle((left, bottom), width, height,fill=False, transform=ax.transAxes, clip_on=False)

      ax.pie(val, colors=colors, pctdistance=0.85, explode=explode)
      ax.text(-1.05, 1.1, f"N {int(val[2]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
      ax.text(-0.1, 1.1, f"U {int(val[1]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
      ax.text(0.65, 1.1, f"S {int(val[0]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
      ax.add_artist(plt.Circle((0, 0), 0.6, color="#00172B"))
      
      return fig

def make_stats_side(df_elo, hist_cols):
      n = len(df_elo['Deck'].unique())
      df_max = df_elo[hist_cols+['Elo']].agg(['idxmax', 'max'])
      df_min = df_elo[df_elo[hist_cols+['Elo']]>0][hist_cols+['Elo']].agg(['idxmin', 'min'])
      # layout
      cols_layout = st.columns([2,2,2,2,3])
      with cols_layout[0]:
            st.metric("Decks in Wertung", value=f"{n}")
             
      with cols_layout[1]:
            st.metric(f"Gesamtzahl Matches", value=f"{int(df_elo['Matches'].sum()/2)}")
             
      with cols_layout[2]:
            st.metric(f"Gesamtzahl Spiele", value=f"{int((df_elo['Siege'].sum()+df_elo['Remis'].sum()+df_elo['Niederlage'].sum())/2)}")
            
      with cols_layout[3]:
            st.metric('Anzahl Spieler', value=len(df_elo['Owner'].unique()))
                  
      with cols_layout[4]:
            pass

      cols_layout = st.columns([3,3,3,3])
      with cols_layout[0]:
            st.metric(f"Höchste Elo", value=f"{df_elo.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']}", delta=f"{int(df_max.loc['max', :].max())}")
                  
      with cols_layout[1]:
            st.metric(f"Aktuelle beste Elo", value=f"{df_elo.at[int(df_max.at['idxmax', 'Elo']), 'Deck']}", delta=f"{int(df_max.at['max', 'Elo'])}")
            
      with cols_layout[2]:
            st.metric(f"Niedrigste Elo", value=f"{df_elo.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']}", delta=f"{int(df_min.loc['min', :].min())}")
            
      with cols_layout[3]:
            st.metric(f"Aktuelle niedrigste Elo", value=f"{df_elo.at[df_min.loc['idxmin', 'Elo'], 'Deck']}", delta=f"{int(df_min.loc['min', 'Elo'].min())}")
            
      st.markdown("----")
      types = list(df_elo['Type'].unique())
      colst = st.columns(len(types))
      for k in range(len(types)):
            with colst[k]:
                  tmp = df_elo[df_elo['Type']==types[k]].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
                  idx = tmp.at[0, 'Deck']
                  d = tmp.at[0, 'Elo']
                  st.metric('Bestes ' + types[k] + ' Deck', value=idx, delta=f"{d}")
      colsk = st.columns(6)
      for i, feat in enumerate(types):
            with colsk[i]:
                  n = len(df_elo[df_elo['Type']==feat])
                  st.metric('Anzahl Decks ' + feat, value=n)
      
      st.markdown("----")
      vergleiche_stile(df_elo)

def get_plyer_infos(df, name, hist_cols):
      cols_spider = ['Attack', 'Control', 'Recovery', 'Consistensy','Combo', 'Resilience']
      n_wp = df[df['Owner']==name]['Wanderpokal'].sum()
      n_fun = df[df['Owner']==name]['Fun Pokal'].sum()
      act_elo_best = df[df['Owner']==name]['Elo'].astype(int).max()
      act_elo_min = df[df['Owner']==name]['Elo'].astype(int).min()
      
      tmp = df[df['Owner']==name][hist_cols+['Elo']].astype(int).max()
      hist_elo_best = int(np.max(tmp[tmp>0]))
      
      tmp = df[df['Owner']==name][hist_cols+['Elo']].astype(int).min()
      hist_elo_min = int(np.min(tmp[tmp>0]))
      
      n_decks = len(df[df['Owner']==name])
      fig = make_spider_plot(np.round(df[df['Owner']==name][cols_spider].mean(), 0).astype(int).values)
      fig2 = make_deck_histo(df[df['Owner']==name])
      fig3 = make_type_histo(df, name)
      return n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min

def make_stats_table(df_elo, ):
      left, right = st.columns([2, 4])
      with left:
            with st.form('filter'):
                  slider_range = st.slider('Stellle den Elo-Bereich ein:', 
                        min_value = int(df_elo['Elo'].min()/10)*10,
                        max_value = int(np.ceil(df_elo['Elo'].max()/10))*10,
                        value = [int(df_elo['Elo'].min()/10)*10, int(np.ceil(df_elo['Elo'].max()/10))*10],
                        step = 10)
                  # filter for players
                  owner = st.multiselect(
                        "Wähle Spieler:",
                        options = df_elo['Owner'].unique(),
                        default = st.session_state['owner'])
                  # filter for showing the data
                  types = st.multiselect(
                        'Wähle Decktyp',
                        options = df_elo['Type'].unique(),
                        default = st.session_state['types_select'])
                  
                  # filter for gruppe
                  tier = st.multiselect(
                        "Wähle die Deckstärke:",
                        options = df_elo['Tier'].unique(),
                        default = st.session_state['tier'])
                        
                  # filter for modus
                  tournament = st.multiselect(
                        'Suche nach einem Turnier:',
                        options = ['Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal'],
                        default = st.session_state['tournament'])
                  st.form_submit_button('Aktualisiere Tabelle')            
      if len(tournament)>0:
            idx = []
            for idx_tour, tour in enumerate(tournament):
                  idx.append(df_elo[df_elo[tour]>0].index.to_numpy())
            idx = np.concatenate(idx)
            df_select = df_elo.loc[idx, :].reset_index(drop=True)
      else:
            df_select = df_elo.copy()
                  
      df_select = df_select.query(
            "Owner == @owner & Tier == @tier & Type == @types"
            )
      df_select = df_select[(df_select['Elo']>=slider_range[0]) & (df_select['Elo']<=slider_range[1])]
      df_select = df_select.reset_index()
      
      with right:
            df_select = df_select[['Deck', 'Elo', '1/4 Differenz', '1/2 Differenz', '1 Differenz', 'Matches', 'Siege', 'Remis', 'Niederlage']].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
            df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']] = df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']].astype(int)
            df_select['Platz'] = df_select.index.to_numpy()+1
            AgGrid(df_select[['Platz', 'Deck', 'Elo', '1/4 Differenz', '1/2 Differenz', '1 Differenz', 'Matches', 'Siege', 'Remis', 'Niederlage']], fit_columns_on_grid_load=True, update_mode='NO_UPDATE',
            theme='streamlit', )

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
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

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
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

def update_elo_ratings(deck1, deck2, erg1, erg2, df_elo, hist_cols, save_cols, db):
      print(deck1)
      print(deck2)
      time.sleep(5)
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
      df_elo.at[idx1, 'dgp'] = (df_elo.at[idx1, 'dgp'] + 2*df_elo.at[idx2, 'Elo'])//3
      df_elo.at[idx2, 'dgp'] = (df_elo.at[idx2, 'dgp'] + 2*df_elo.at[idx1, 'Elo'])//3
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
      
      key1 = df_elo.at[idx1, 'key']
      key2 = df_elo.at[idx2, 'key']
      
      update_element(key1, df_elo.loc[idx1, :], save_cols, hist_cols, db)
      update_element(key2, df_elo.loc[idx2, :], save_cols, hist_cols, db)
      
def vergleiche_die_decks(df_elo, hist_cols, deck):
      plot, stats = st.tabs(['Elo-Verlauf', 'Statistiken'])
      with plot:
            with st.container():
                  fig = plot_elo_history(df_elo, deck, hist_cols)
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
                              
                              idx = int(df_elo[df_elo['Deck']==deck[counter]].index.to_numpy())
                              values = df_elo.loc[idx, ['Siege', 'Remis', 'Niederlage']].to_list()
                              
                              with columns[2*i]:
                                    st.header(df_elo.at[idx, 'Deck']+" :star:"*int(df_elo.at[idx, 'Wanderpokal']))
                                    st.subheader(df_elo.at[idx, 'Tier'])
                                    st.subheader(f"Matches: {int(df_elo.at[idx, 'Matches'])}")
                                    st.subheader(f"Spiele: {int(np.sum(values))}")
                                    st.metric('Aktuelle Elo', df_elo.at[idx, 'Elo'], int(df_elo.at[idx, '1/4 Differenz']))
                                    st.metric('Gegner Stärke', int(df_elo.at[idx, 'dgp']))
                                    st.caption('Wanderpokal:   ' + ':star:'*int(df_elo.at[idx, 'Wanderpokal']))
                                    st.caption('Fun Pokal:     ' + ':star:'*int(df_elo.at[idx, 'Fun Pokal']))
                                    st.caption('Meisterschaft: ' + ':star:'*int(df_elo.at[idx, 'Meisterschaft']))
                                    st.caption('Liga Pokal:    ' + ':star:'*int(df_elo.at[idx, 'Liga Pokal']))
                                    
                              with columns[2*i+1]:
                                    st.header(" ")
                                    st.header(" ")
                                    
                                    fig = semi_circle_plot(values)
                                    st.pyplot(fig, transparent=True)
                                    
                                    categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
                                    fig = make_spider_plot(df_elo.loc[idx, categories].astype(int).to_numpy())
                                    st.pyplot(fig, transparent=True)
                                    
                                    counter += 1
    
def alles_zu_einem_deck(deck_i, df_elo, hist_cols):
      if len(deck_i) > 0:
            # calculations
            idx_deck = int(df_elo[df_elo['Deck']==deck_i].index.to_numpy())
            tmp_df = df_elo.copy()
            tmp_df = tmp_df[['Deck', 'Elo']].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
            platz = int(tmp_df[tmp_df['Deck']==deck_i].index.to_numpy())+1
            percentage = int(platz/len(df_elo)*1000)/10
            # Layout
            st.header(df_elo.at[idx_deck, 'Tier']+ '     ' + df_elo.at[idx_deck, 'Type'])
            
            deck_cols = st.columns([1,1,1,1,1,1.5,2])      
            with deck_cols[0]:
                  st.caption(f"Aktueller Rang: {platz}")
                  st.caption(f"Obere {percentage}% aller Decks")
                  st.caption('Wanderpokal:   ' + ':star:'*int(df_elo.at[idx_deck, 'Wanderpokal']))
                  st.caption('Fun Pokal:     ' + ':star:'*int(df_elo.at[idx_deck, 'Fun Pokal']))
                  st.caption('Meisterschaft: ' + ':star:'*int(df_elo.at[idx_deck, 'Meisterschaft']))
                  st.caption('Liga Pokal:    ' + ':star:'*int(df_elo.at[idx_deck, 'Liga Pokal']))
                  
            with deck_cols[1]:
                  st.metric(f"Beste Elo", value=f"{int(df_elo.loc[idx_deck, hist_cols+['Elo']].max())}")
                  st.metric(f"Schlechteste Elo", value=f"{int(df_elo.loc[idx_deck, hist_cols+['Elo']].min())}")

            with deck_cols[2]:
                  st.metric('Aktuelle Elo', df_elo.at[idx_deck, 'Elo'], int(df_elo.at[idx_deck, '1/4 Differenz']))
                  st.metric('Gegner Stärke', int(df_elo.at[idx_deck, 'dgp']))
                   
            with deck_cols[3]:
                  st.metric('Jahresmittel', value=int(df_elo.at[idx_deck, 'Mean 1 Year']), delta=int(df_elo.at[idx_deck, 'Std. 1 Year']))
                  st.metric('Jahresänderung', value=int(df_elo.loc[idx_deck, '1 Differenz']))

            with deck_cols[4]:
                  st.metric(f"Matches", value=f"{int(df_elo.at[idx_deck, 'Matches'])}")
                  st.metric(f"Spiele", value=f"{int(df_elo.at[idx_deck, 'Siege']+df_elo.at[idx_deck, 'Remis']+df_elo.at[idx_deck, 'Niederlage'])}")
                  
            with deck_cols[5]:
                  fig = semi_circle_plot(df_elo.loc[idx_deck, ['Siege', 'Remis', 'Niederlage']].to_list())
                  st.pyplot(fig, transparent=True)
            with deck_cols[6]:
                  categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
                  fig = make_spider_plot(df_elo.loc[idx_deck, categories].astype(int).to_numpy())
                  st.pyplot(fig, transparent=True)

            c1, c2 = st.columns([3,2])
            with c1:
                  fig, axs = plt.subplots(1,1, figsize=(5, 3))
                  axs.plot(range(5), df_elo.loc[idx_deck, hist_cols[-4:]+['Elo']].to_numpy())
                  axs.set_title('Eloverlauf des letzen Jahres', color='white')
                  axs.set_xticks(range(5))
                  axs.set_xticklabels(hist_cols[-4:]+['Elo'], color='white', rotation=45)
                  axs.set_ylabel('Elo', color="white")
                  axs.grid()
                  axs.spines['bottom'].set_color('white')
                  axs.spines['left'].set_color('white')
                  axs.tick_params(colors='white', which='both')
                  st.pyplot(fig, transparent=True)
            with c2:      
                  pass
         
            
def vergleiche_die_spieler(df_elo, hist_cols):
      player_cols = st.columns(5)
      # Chriss
      with player_cols[0]:
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Christoph', hist_cols)
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
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Finn', hist_cols)
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
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Frido', hist_cols)
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
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Jan', hist_cols)
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
            n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Thomas', hist_cols)
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
      
def vergleiche_stile(df_elo):
      histo, viobox, c3 = st.columns([1, 1, 1])
      types = df_elo['Type'].unique()
      with histo:
            n_types = len(types)
            fig, axs = plt.subplots(1, figsize=(8,5))
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
            fig, axs = plt.subplots(1, figsize=(8,5))
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
      
      with c3:
            fig = plot_deck_desity(np.array(df_elo['Elo'].values).squeeze())
            st.pyplot(fig, transparent=True)
      
@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_deck_desity(values):
      
      Label = ['Kartenstapel','Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0']
      H, bins = do_histogram(values, b=10, d=False)
      
      mean = np.mean(values)
      std = np.std(values)
      
      x = np.linspace(mean-4*std, mean+4*std, 500)
      a = 0.3

      fig = plt.figure(figsize=(8,8))
      fig.set_figheight(10)
      fig.set_figwidth(15)
      
      # create grid for different subplots
      spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0, height_ratios=[6, 1])
      axs = fig.add_subplot(spec[0])
      axs.set_title('Elo-Verteilung', fontsize=20, color='white')
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
      axs.tick_params(axis='y', labelsize=15, color='white') 
      return fig