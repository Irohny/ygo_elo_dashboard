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

def make_spider_plot(df):
      categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
      
      angles = [n / float(6) * 2 * np.pi for n in range(6)]
      angles += angles[:1]
      angles = (np.array(angles) + np.pi/2)%(2*np.pi)
      # Initialise the spider plot
      fig, axs = plt.subplots(figsize=(1,1), subplot_kw=dict(polar=True))
      # Draw one axe per variable + add labels
      plt.xticks(angles[:-1], categories, color='white', size=4)
      
 
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

st.set_page_config(page_title='YGO-Elo-Dashboard', page_icon=':trophy:' ,layout='wide')
# ---- HIDE STREAMLIT STYLE ----

st.title('Yu-Gi-Oh! Elo-Dashbord')

@st.cache
def fetch_and_clean_data():
       df_elo = pd.read_csv('./web_app_table.csv')
       # process data
       cols = df_elo.columns
       df_elo['Elo'] = df_elo[cols[-1]].astype(int)
       df_elo['1/4 Differenz'] = df_elo['Elo']-df_elo[cols[-2]]
       df_elo['1/4 Differenz'] = df_elo['1/4 Differenz'].fillna(0).astype(int)
       df_elo['1/2 Differenz'] = df_elo['Elo']-df_elo[cols[-3]]
       df_elo['1/2 Differenz'] = df_elo['1/2 Differenz'].fillna(0).astype(int)
       df_elo['1 Differenz'] = df_elo['Elo']-df_elo[cols[-5]]
       df_elo['1 Differenz'] = df_elo['1 Differenz'].fillna(0).astype(int)
       df_elo['Mean 1 Year'] = np.mean(df_elo[cols[-5:-1]],1)
       df_elo['Mean 1 Year'] = df_elo['Mean 1 Year'].fillna(0).astype(int)
       df_elo['Std. 1 Year'] = np.std(df_elo[cols[-5:-1]],1)
       df_elo['Std. 1 Year'] = df_elo['Std. 1 Year'].fillna(0).astype(int)
       return df_elo.fillna(0)
 
# build data
df_elo = fetch_and_clean_data()
df_select = df_elo.copy()
cols = df_elo.columns
vgl_decks, vgl_player, vgl_style, ges_stats = st.tabs(['Vergleiche die Decks', 'Vergleiche die Spieler', 'Vergleiche die Stile','Gesamtstatistik'])
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
                        tmp = np.append(tmp, df_select.loc[idx, cols[14:-6]].to_numpy())
                        axs.plot(df_select.loc[idx, cols[14:-6]], label=df_select.at[idx, 'Deck'], lw=3)
                        
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
                        columns = st.columns(4)
                        for i in range(4):
                              with columns[i]:
                                    if counter >= len(deck):
                                          break
                                    
                                    idx = int(df_select[df_select['Deck']==deck[counter]].index.to_numpy())
                                    st.header(df_select.at[idx, 'Deck']+" :star:"*int(df_select.at[idx, 'Wanderpokal']))
                                    st.subheader(df_select.at[idx, 'Tier'])
                                    st.metric('Aktuelle Elo', df_select.at[idx, 'Elo'], int(df_select.at[idx, '1/4 Differenz']))
                                    categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
                                    fig = make_spider_plot(df_select.loc[idx, categories].astype(int).to_numpy())
                                    st.pyplot(fig, transparent=True)
                                    st.caption('Wanderpokal:   ' + ':star:'*int(df_select.at[idx, 'Wanderpokal']))
                                    st.caption('Fun Pokal:     ' + ':star:'*int(df_select.at[idx, 'Fun Pokal']))
                                    st.caption('Meisterschaft: ' + ':star:'*int(df_select.at[idx, 'Meisterschaft']))
                                    st.caption('Liga Pokal:    ' + ':star:'*int(df_select.at[idx, 'Liga Pokal']))
                                    
                                    counter += 1
      
def get_plyer_infos(df, name, cols):
      cols_spider = ['Attack', 'Control', 'Recovery', 'Consistensy','Combo', 'Resilience']
      n_wp = df[df['Owner']==name]['Wanderpokal'].sum()
      n_fun = df[df['Owner']==name]['Fun Pokal'].sum()
      act_elo_best = df[df['Owner']==name]['Elo'].astype(int).max()
      act_elo_min = df[df['Owner']==name]['Elo'].astype(int).min()
      
      tmp = df[df['Owner']==name][cols[14:-6]].astype(int).max()
      hist_elo_best = int(np.max(tmp[tmp>0]))
      
      tmp = df[df['Owner']==name][cols[14:-6]].astype(int).min()
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
      chris, finn, frido, jan = st.columns(4)
      with chris:
            n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Christoph', cols)
            st.header('Chirstoph ')
            st.text(f'Decks in Wertung: {n_decks}')
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)
            
      with finn:
            n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Finn', cols)
            st.header('Finn')
            st.text(f'Decks in Wertung: {n_decks}')
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)
            
      with frido:
            n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Frido', cols)
            st.header('Frido')
            st.text(f'Decks in Wertung: {n_decks}')
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
            st.pyplot(fig, transparent=True)
            st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
            st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
            st.pyplot(fig2, transparent=True)
            st.pyplot(fig3, transparent=True)
            
      with jan:
            n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = get_plyer_infos(df_elo, 'Jan', cols)
            st.header('Jan')
            st.text(f'Decks in Wertung: {n_decks}')
            st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
            st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
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
            df_max = df_elo[cols[16:-6]].agg(['idxmax', 'max'])
            df_min = df_elo[df_elo[cols[16:-6]]>0][cols[16:-6]].agg(['idxmin', 'min'])
            st.subheader(f"     Decks in Wertung {n}")
            st.subheader('')
            st.subheader(f"     Höchste Elo:             {int(df_max.loc['max', :].max())}, {df_elo.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']}")
            st.subheader(f"     Aktuelle beste Elo:      {int(df_max.at['max', cols[-7]])}, {df_elo.at[int(df_max.at['idxmax', cols[-7]]), 'Deck']}")
            st.subheader('')
            st.subheader(f"     Niedrigste Elo:          {int(df_min.loc['min', :].min())}, {df_elo.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']}")
            st.subheader(f"     Aktuelle niedrigste Elo: {int(df_min.loc['min', cols[-7]].min())}, {df_elo.at[df_min.loc['idxmin', cols[-7]], 'Deck']}")
            
            
            n_decks = len(df_elo['Deck'].unique())
            
            
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