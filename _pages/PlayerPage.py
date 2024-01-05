import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from emoji import emojize
import plotly.graph_objects as go
from pages._pages.Visualization import Visualization

class PlayerPage:
    '''
    Class for displaying player stats
    '''
    def __init__(self, df, hist_cols):
        self.back_color = "#00172B"
        self.filed_color = "#0083B8"
        self.vis = Visualization()
        self.__build_page_layout(df, hist_cols)

    def __build_page_layout(self, df, hist_cols):
        '''
        '''
        player_cols = st.columns(5)
        # Chriss
        with player_cols[0]:
            self.__display_player_stats('Christoph', df, hist_cols)                
        # Finn
        with player_cols[1]:
            self.__display_player_stats('Finn', df, hist_cols)        
        # Frido
        with player_cols[2]:
            self.__display_player_stats('Frido', df, hist_cols)
        # Jan
        with player_cols[3]:
            self.__display_player_stats('Jan', df, hist_cols)                   
        # Thomas
        with player_cols[4]:
            self.__display_player_stats('Thomas', df, hist_cols)

    def __display_player_stats(self, name, df, hist_cols):
        '''
        '''
        n_wp, n_decks, n_fun, n_local_win, n_local_top, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = self.__get_player_infos(df, name, hist_cols)
        st.header(name)
        tmp = df[df['Owner']==name][['Siege', 'Remis', 'Niederlage']].sum().to_list()
        fig = self.vis.plotly_gauge_plot(100*tmp[0]/sum(tmp)//1)#self.__semi_circle_plot(tmp.values)
        st.metric(f"Matches", f"{int(df[df['Owner']==name]['Matches'].sum()//2)}", f"{int(sum(tmp)//2)} Spiele")
        st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
        st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
        df_stats = pd.DataFrame(columns=['Decks', n_decks], index=range(4))
        df_stats.at[0, 'Decks'] = 'Wanderpokal'
        df_stats.at[0, n_decks] = emojize((':trophy:'*int(n_wp)))
        df_stats.at[1, 'Decks'] = 'Fun Pokal'
        df_stats.at[1, n_decks] = emojize((':star:'*int(n_fun)))
        df_stats.at[2, 'Decks'] = 'Local Top'
        df_stats.at[2, n_decks] = emojize((':star:'*int(n_local_top)))
        df_stats.at[3, 'Decks'] = 'Local Win'
        df_stats.at[3, n_decks] = emojize((':star:'*int(n_local_win)))
        st.dataframe(df_stats, hide_index=True, use_container_width=True)
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        st.plotly_chart(fig,theme="streamlit", use_container_width=True)
        st.pyplot(fig2, transparent=True)
        st.pyplot(fig3, transparent=True)
    
    def __get_player_infos(self, df, name, hist_cols):
        '''
        Method for KPI estimation by given player
        :param df: dataframe with all data
        :param name: name of the player
        :param hist_cols: list with historic elo ratings 
        :return:
        '''
        cols_spider = ['Attack', 'Control', 'Recovery', 'Consistensy','Combo', 'Resilience']
        n_wp = 0
        n_local_win = 0
        n_fun = 0
        n_local_top = 0
        for idx in df[df['Owner']==name].index.to_numpy():
            n_wp += int(df.at[idx, 'Meisterschaft']['Win']['Wanderpokal'])
            n_fun += int(df.at[idx, 'Meisterschaft']['Win']['Fun Pokal'])
            n_local_win += int(df.at[idx, 'Meisterschaft']['Win']['Local'])
            n_local_top += int(df.at[idx, 'Meisterschaft']['Top']['Local'])
        act_elo_best = df[df['Owner']==name]['Elo'].astype(int).max()
        act_elo_min = df[df['Owner']==name]['Elo'].astype(int).min()
        
        tmp = df[df['Owner']==name][hist_cols+['Elo']].astype(int).max()
        hist_elo_best = int(np.max(tmp[tmp>0]))
        
        tmp = df[df['Owner']==name][hist_cols+['Elo']].astype(int).min()
        hist_elo_min = int(np.min(tmp[tmp>0]))
        
        n_decks = len(df[df['Owner']==name])
        fig = self.vis.make_spider_plot(list(np.round(df[df['Owner']==name][cols_spider].mean(), 0).astype(int).values))
        fig2 = self.__make_deck_histo(df[df['Owner']==name])
        fig3 = self.__make_type_histo(df, name)
        return n_wp, n_decks, n_fun, n_local_win, n_local_top, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min

    def __make_deck_histo(self, df):
        '''
        Method for creating a bar plot with number of decks in a tier range
        :param df: dataframe a data
        :return: matplotlib figure object 
        '''
        n1 = len(df[df['Tier']=='Kartenstapel'])
        n2 = len(df[df['Tier']=='Fun'])
        n3 = len(df[df['Tier']=='Good'])
        n4 = len(df[df['Tier']=='Tier 2'])
        n5 = len(df[df['Tier']=='Tier 1'])
        n6 = len(df[df['Tier']=='Tier 0'])
        
        fig, axs = plt.subplots(1, figsize=(9,9))
        axs.bar([1,2,3,4,5,6], [n1, n2, n3, n4, n5, n6], color='gray')
        axs.set_xticks([1,2,3,4,5,6])
        axs.set_xticklabels(['Kartenstapel', 'Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0'], rotation=-45, color='black')
        axs.set_ylabel('Anzahl Decks', color='black')
        axs.grid()
        axs.set_title('Verteilung der Decks auf die Spielstärken', color='black')
        axs.spines['bottom'].set_color('black')
        axs.spines['left'].set_color('black')
        axs.tick_params(colors='black', which='both')
        return fig

    def __make_type_histo(self, df, name):
        '''
        Method for creating a bar plot with deck types for a given player
        :param df: dataframe with all data
        :param name: name of the player
        :return fig: matplotlib figure object
        '''
        types = df['Type'].unique()
        n_types = len(types)
        fig, axs = plt.subplots(1, figsize=(8,8))
        n_decks = np.zeros(n_types)
        for k in range(n_types):
                n_decks[k] = len(df[(df['Type']==types[k])&(df['Owner']==name)])
        axs.bar(range(n_types), n_decks, color='gray')
        axs.set_xticks(range(n_types))
        axs.set_xticklabels(types, rotation=-45, color='black')
        axs.set_ylabel('Anzahl Decks', color='black')
        axs.grid()
        axs.set_title('Verteilung der Decks auf die Typen', color='black')
        axs.spines['bottom'].set_color('black')
        axs.spines['left'].set_color('black')
        axs.tick_params(colors='black', which='both')
        return fig
