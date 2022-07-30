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
import datetime

#@st.cache
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

@st.cache      
def fetch_and_clean_data(df):
      #df_elo = pd.read_csv('./web_app_table.csv')
      # process data
      cols_to_int = ['Elo', 'dgp', 'Wanderpokal', 'Meisterschaft', 'Liga Pokal', 'Fun Pokal', 'Attack', 'Combo', 'Resilience', 'Recovery',
                        'Consistensy', 'Control','Matches', 'Siege', 'Remis', 'Niederlage']
      for k in cols_to_int:
            df[k] = df[k].astype(float)
      cols = df.columns.to_list()
      df['Elo'] = df['Elo'].astype(int)
      df['1/4 Differenz'] = df['Elo']-df[cols[-1]]
      df['1/4 Differenz'] = df['1/4 Differenz'].fillna(0).astype(int)
      df['1/2 Differenz'] = df['Elo']-df[cols[-2]]
      df['1/2 Differenz'] = df['1/2 Differenz'].fillna(0).astype(int)
      df['1 Differenz'] = df['Elo']-df[cols[-4]]
      df['1 Differenz'] = df['1 Differenz'].fillna(0).astype(int)
      df['Mean 1 Year'] = np.mean(df[cols[-5:-1]+['Elo']],1)
      df['Mean 1 Year'] = df['Mean 1 Year'].fillna(0).astype(int)
      df['Std. 1 Year'] = np.std(df[cols[-5:-1]+['Elo']],1)
      df['Std. 1 Year'] = df['Std. 1 Year'].fillna(0).astype(int)
       
      return df.fillna(0)

@st.cache
def sort_hist_cols(hist_cols):
      t = []
      for c in hist_cols:
          t.append(datetime.date(year = int(c[-4:]), month = int(c[:2]), day=1))
      t = pd.Series(t).sort_values().reset_index(drop=True)
      out = []
      for i in range(len(t)):
            out.append(t[i].strftime('%m/%Y'))
      return out

@st.cache
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

@st.cache
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

@st.cache
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

@st.cache
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

@st.cache      
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
      
      key1 = df_elo.at[idx1, 'key']
      key2 = df_elo.at[idx2, 'key']
      
      update_element(key1, df_elo.loc[idx1, :], save_cols, hist_cols, db)
      update_element(key2, df_elo.loc[idx2, :], save_cols, hist_cols, db)
