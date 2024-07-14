import streamlit as st
import pandas as pd
import numpy as np

from DataModel.utils.connect2deta import connect2deta

class TournamentModel:
    def __init__(self, load_data=True):
        self.db = connect2deta(key=st.secrets['ygo_tournament_key'],
                                  name="YuGiOh_Tournaments")
        if load_data:
            # get tournmaent data
            self.df = pd.DataFrame(self.db.fetch().items)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.__group_results()
            self.__calculate_tournament_score()

    def get(self,):
        return self.df, self.df_agg, self.df_score
    
    def __group_results(self, )->None:
        """
        Method for combining torunament dat to deck data for local results
        """
        df_agg = []
        for tour in ['Fun', 'Wanderpokal', 'Local', 'Regional']:
            for res in ['Teilnahme', 'Top', 'Win']:
                filtered = self.__group_results_for_tournament(res, tour)
                if filtered.empty:
                    continue
                df_agg.append(filtered)
        self.df_agg = pd.concat(df_agg, ignore_index=True)

    def __group_results_for_tournament(self, result:pd.DataFrame, mode:str):
        """
        Method for merging tournament data to deck data
        :param result: tournament satnding
        :param new_name: name of the tournament standing column
        :return result:merged dataframe
        """
        group_cols = ['Deck', 'Win', 'Draw', 'Loss', 'key']
        # get filter for input 
        filtered = self.df[(self.df['Standing'] == result) & (self.df['Mode'] == mode)]
        if filtered.empty:
            return pd.DataFrame()
        filtered = filtered[group_cols].groupby('Deck').agg({'Win':sum, 'Draw':sum,
                                                           'Loss':sum, 'key':'count'}).reset_index()
        filtered.rename(columns={'key':'Anzahl'}, inplace=True)
        filtered['Standing'] = result
        filtered['Mode'] = mode 
        return filtered
    
    def insert_tournament(self, result_dict):
        '''
        Method for updating the tournament column of a spezific deck
        :param result_dict: diconary with tournament results
        '''
        self.db.put(result_dict)		
    
    def __calculate_tournament_score(self, ):
        # tournament standings
        self.df_score = self.df[['Deck', 'Win', 'Loss', 'Draw', 'Standing']].copy()
        self.df_score['Turniere'] = 0
        self.df_score['Top-Rate'] = 0
        self.df_score['Match-Win-Rate'] = 0
        self.df_score['Standing'] = self.df_score['Standing'].apply(self.__tournament_points)
        self.df_score['Points'] = (3*self.df_score['Win'] + self.df_score['Draw'] + self.df_score['Standing'])
        self.df_score = self.df_score.groupby(by='Deck').sum()
        
        tops = self.df[self.df['Standing']=='Top'].groupby(by='Deck').count()
        counts = self.df[['Standing', 'Deck']].groupby(by='Deck').count()
        
        for idx in self.df_score.index:
            n = counts.at[idx, 'Standing']
            if n==0:
                continue
            if idx in tops.index:
                self.df_score.at[idx, 'Top-Rate'] = np.round(100*tops.at[idx, 'Standing']/n,2)
            self.df_score.at[idx, 'Turniere'] = n

        self.df_score['Points'] /= self.df_score['Turniere']
        self.df_score['Points'] = np.round(self.df_score['Points'], 2)

        total_tourn_games = self.df_score['Win']+self.df_score['Draw']+self.df_score['Loss']
        self.df_score['Match-Win-Rate'] = self.df_score['Win']/total_tourn_games
        self.df_score['Match-Win-Rate'] = np.round(self.df_score['Match-Win-Rate'], 2)
        
        self.df_score.sort_values(by='Points', ascending=False, inplace=True)
        self.df_score.reset_index(inplace=True)
        self.df_score['Platz'] = self.df_score.index.to_numpy()+1
        self.df_score.rename(columns={'Win':'Tourn_Win', 'Loss':'Tourn_Loss', 'Draw':'Tourn_Draw'},
                             inplace=True)
        
    def __tournament_points(self, x):
        """
        """
        coding = {'Win':10, 'Top':5, 'Teilnahme':1}
        return coding[x]