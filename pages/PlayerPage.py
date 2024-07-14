import streamlit as st
import numpy as np
import pandas as pd
from emoji import emojize
from VisualizationTools.Visualization import Visualization

class PlayerPage:
    '''
    Class for displaying player stats
    '''
    def __init__(self):
        
        self.back_color = "#00172B"
        self.filed_color = "#0083B8"
        self.vis = Visualization()
        self.__build_page_layout(st.session_state['deck_data'].copy(), 
                                 st.session_state['history_columns'])

    def __build_page_layout(self, df, hist_cols):
        '''
        Method for creating the plyer page layout
        '''
        inputs = st.columns(3)
        players = inputs[0].multiselect('Spieler:', options=['Christoph', 'Frido', 'Jan', 'Thomas', 'Finn', 'Nicolas'],
                                default=['Christoph', 'Frido', 'Jan', 'Thomas'])
        # skip processing if no player is selected
        if not players:
            return
        
        player_cols = st.columns(len(players))
        # Chriss
        for idx, player in enumerate(players):
            self.__display_player_stats(player, df, hist_cols, player_cols[idx])                
        
    def __display_player_stats(self, name, df, hist_cols, st_obj):
        '''
        Method for creating a player stats aggregation
        :param name: name of the player
        :param df: dataframe with decks
        :param hist_cols: history columns of df
        :param st_obj: streamlit object for layout
        '''
        # get player stats
        n_wp, n_decks, n_fun, n_local_win, n_local_top, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = self.__get_player_infos(df, name, hist_cols)
        # display player stats
        st_obj.header(name)
        tmp = df[df['Owner']==name][['Siege', 'Remis', 'Niederlage']].sum().to_list()
        fig = self.vis.plotly_gauge_plot(100*tmp[0]/sum(tmp)//1)
        st_obj.metric("Matches", f"{int(df[df['Owner']==name]['Matches'].sum()//2)}", f"{int(sum(tmp)//2)} Spiele")
        st_obj.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
        st_obj.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
        df_stats = pd.DataFrame(columns=['Decks', n_decks], index=range(4))
        df_stats.at[0, 'Decks'] = 'Wanderpokal'
        df_stats.at[0, n_decks] = emojize((':trophy:'*int(n_wp)))
        df_stats.at[1, 'Decks'] = 'Fun Pokal'
        df_stats.at[1, n_decks] = emojize((':star:'*int(n_fun)))
        df_stats.at[2, 'Decks'] = 'Local Top'
        df_stats.at[2, n_decks] = emojize((':star:'*int(n_local_top)))
        df_stats.at[3, 'Decks'] = 'Local Win'
        df_stats.at[3, n_decks] = emojize((':star:'*int(n_local_win)))
        st_obj.dataframe(df_stats, hide_index=True, use_container_width=True)
        st_obj.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        st_obj.plotly_chart(fig,theme="streamlit", use_container_width=True)
        st_obj.plotly_chart(fig2,theme="streamlit", use_container_width=True)
        st_obj.plotly_chart(fig3,theme="streamlit", use_container_width=True)
    
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
        tmp = df[df['Owner']==name].groupby('Tier')['Deck'].count().reset_index()
        tmp = self.__order_strings(tmp, 'Tier', ['Tier 0', 'Tier 1', 'Tier 2', 'Good', 'Fun', 'Kartenstapel'])
        fig2 = self.vis.ploty_bar(tmp, 'Tier', 'Deck', False, title='Deck Tiers')
        tmp = df[df['Owner']==name].groupby('Type')['Deck'].count().reset_index()
        fig3 = self.vis.ploty_bar(tmp, 'Type', 'Deck', False, title='Decktypen:')
        #fig3 = self.__make_type_histo(df, name)
        return n_wp, n_decks, n_fun, n_local_win, n_local_top, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min

    def __order_strings(self, df, col, order):
        res = []
        values = list(df[col].unique())
        for tag in order:
            if tag not in values:
                continue
            idx = df[df[col]==tag].index
            res.append(df.loc[idx])
        return pd.concat(res, ignore_index=True)
            