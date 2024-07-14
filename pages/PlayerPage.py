import streamlit as st
import numpy as np
import pandas as pd
from emoji import emojize
import VisualizationTools as vit

class PlayerPage:
    '''
    Class for displaying player stats
    '''
    def __init__(self):
        self.df = st.session_state['deck_data'].copy()
        self.hist_cols = st.session_state['history_columns']
        self.tournament = st.session_state['tournament_data'].copy()
        self.cols_spider = ['Attack', 'Control', 'Recovery', 'Consistensy','Combo', 'Resilience']
        
        self.vis = vit.Visualization()
        self.__build_page_layout()

    def __build_page_layout(self):
        '''
        Method for creating the plyer page layout
        '''
        st.title(':trophy: Spieler :trophy:', anchor='anchor_tag')
        inputs = st.columns(3)
        players = inputs[0].multiselect('Spieler:', options=['Christoph', 'Frido', 'Jan', 'Thomas', 'Finn', 'Nicolas'],
                                default=['Christoph', 'Frido', 'Jan', 'Thomas'], label_visibility='collapsed')
        # skip processing if no player is selected
        if not players:
            return
        
        player_cols = st.columns(len(players))
        # Chriss
        for idx, player in enumerate(players):
            self.__display_player_stats(player, player_cols[idx].container(border=True))                
        
    def __display_player_stats(self, name:str, st_obj:st)->None:
        '''
        Method for creating a player stats aggregation
        :param name: name of the player
        :param df: dataframe with decks
        :param hist_cols: history columns of df
        :param st_obj: streamlit object for layout
        '''
        df_player = self.df[self.df['Owner']==name].reset_index(drop=True)
        df_tour = self.tournament[self.tournament['Deck'].isin(df_player['Deck'].values)].reset_index(drop=True)
        # get player stats
        pokale = self.__get_player_infos(df_player, df_tour)
        # display player stats
        st_obj.header(name)
        subcols = st_obj.columns(2)

        tmp = df_player[self.hist_cols+['Elo']].astype(int).max()
        hist_elo_best = int(np.max(tmp[tmp>0]))
        
        subcols[0].metric('Decks', len(df_player), delta=f"Aktiv {sum(df_player['active'])}")
        subcols[0].metric('Beste Elo', value=f"{df_player['Elo'].max()}", 
                          delta=f"Insgesamt {hist_elo_best}", delta_color='off')
        
        tmp = df_player[self.hist_cols+['Elo']].astype(int).min()
        hist_elo_min = int(np.min(tmp[tmp>0]))
        
        games = df_player['Siege'].sum()+df_player['Remis'].sum()+df_player['Niederlage'].sum()
        subcols[1].metric("Matches", f"{int(df_player['Matches'].sum())}", f"{games} Spiele")
        subcols[1].metric('Schlechteste Elo', value=f"{df_player['Elo'].min()}", 
                          delta=f"Insgesamt {hist_elo_min}", delta_color='off')
        
        n_tours = len(df_tour)
        top_rate = (pokale['Local Top'] + pokale['Local Win'])/n_tours
        col1, cols2 = f"Turniere {n_tours}", f"Top-Rate {int(100*top_rate)}%" 
        df_stats = pd.DataFrame(columns=[col1, cols2], index=range(4))
        df_stats.at[0, col1] = 'Wanderpokal'
        df_stats.at[0, cols2] = emojize((':trophy:'*int(pokale["Wanderpokal"])))
        df_stats.at[1, col1] = 'Fun Pokal'
        df_stats.at[1, cols2] = emojize((':star:'*int(pokale["Funpokal"])))
        df_stats.at[2, col1] = 'Local Top'
        df_stats.at[2, cols2] = emojize((':star:'*int(pokale["Local Top"])))
        df_stats.at[3, col1] = 'Local Win'
        df_stats.at[3, cols2] = emojize((':star:'*int(pokale["Local Win"])))
        
        st_obj.dataframe(df_stats, hide_index=True, use_container_width=True)
        tabs = st_obj.tabs(['Eigenschaften', 'Gewinrate', 'Tiers', 'Decktypen', 'Turnier'])
        # Eigenschaften
        spiderplot = self.vis.make_spider_plot(list(df_player[self.cols_spider].mean().astype(int).values))
        tabs[0].plotly_chart(spiderplot, theme="streamlit", use_container_width=True)
        # Win Rate
        tmp = df_player[['Siege', 'Remis', 'Niederlage']].sum().to_list()
        win_rate = self.vis.plotly_gauge_plot(100*tmp[0]/sum(tmp)//1)
        tabs[1].plotly_chart(win_rate,theme="streamlit", use_container_width=True)
        # Tiers
        tmp = df_player.groupby('Tier')['Deck'].count().reset_index()
        tmp = self.__order_strings(tmp, 'Tier', ['Tier 0', 'Tier 1', 'Tier 2', 'Good', 'Fun', 'Kartenstapel'])
        tier_plot = self.vis.ploty_bar(tmp, 'Tier', 'Deck', False)
        tabs[2].plotly_chart(tier_plot,theme="streamlit", use_container_width=True)
        # Deckkategorien
        tmp = df_player.groupby('Type')['Deck'].count().reset_index()
        categorie_histogram = self.vis.ploty_bar(tmp, 'Type', 'Deck', False)
        tabs[3].plotly_chart(categorie_histogram,theme="streamlit", use_container_width=True)
        # turnier win rate
        tmp = [df_tour['Win'].sum(), df_tour['Draw'].sum(), df_tour['Loss'].sum()]
        win_rate = self.vis.plotly_gauge_plot(100*tmp[0]/sum(tmp)//1)
        tabs[4].plotly_chart(win_rate,theme="streamlit", use_container_width=True)
        
    
    def __get_player_infos(self, df:pd.DataFrame, df_tour:pd.DataFrame)->dict:
        '''
        Method for KPI estimation by given player
        :param df: dataframe with all data
        :param name: name of the player
        :param hist_cols: list with historic elo ratings 
        :return:
        '''
        pokale = {'Wanderpokal':0,'Funpokal':0}
        pokale['Local Win'] = sum((df_tour['Mode']=='Local')&(df_tour['Standing']=='Win'))
        pokale['Local Top'] = sum((df_tour['Mode']=='Local')&(df_tour['Standing']=='Top'))
        for idx in df.index.to_numpy():
            pokale['Wanderpokal'] += int(df.at[idx, 'Meisterschaft']['Win']['Wanderpokal'])
            pokale['Funpokal'] += int(df.at[idx, 'Meisterschaft']['Win']['Fun Pokal'])
                
        return pokale

    def __order_strings(self, df, col, order):
        res = []
        values = list(df[col].unique())
        for tag in order:
            if tag not in values:
                continue
            idx = df[df[col]==tag].index
            res.append(df.loc[idx])
        return pd.concat(res, ignore_index=True)
            