from deta import Deta
import numpy as np
import pandas as pd
import datetime
import streamlit as st

class datamodel:
	'''
	Dataclass model for getting and processing data for visualization
	'''
	def __init__(self, ):
		# create database object
		deta = Deta(st.secrets['key'])
		deta2 = Deta(st.secrets['ygo_tournament_key'])
		# create databse connection
		self.db = deta.Base("YgoEloBase")
		self.tourn_db = deta2.Base("YuGiOh_Tournaments")
		# set default miesteschaft dictonary for saving and updating
		self.df_default_meisterschaft = pd.DataFrame(index=['Wanderpokal', 'Local', 'Fun Pokal'], columns=['Teilnahme', 'Top', 'Win']).fillna(0).to_dict()
		# get tournmaent data
		self.tdf = pd.DataFrame(self.tourn_db.fetch().items)
		self.tdf['Date'] = pd.to_datetime(self.tdf['Date'])
		# load data to session state of get data from session state
		if "reload_flag" not in st.session_state or st.session_state['reload_flag']:
			# get data from data base
			self.df, self.cols, self.hist_cols = self.__create_data_model()
			# setup session state
			st.session_state = {'owner':self.df['Owner'].unique(), 
								'types_select':self.df['Type'].unique(),
								'tier':self.df['Tier'].unique(),
								'sorting':['Elo-Rating'],
								'tournament':'Alle',
								'deck_i':'',
								'deck':[],
								'reload_flag':False,
								'dataframe':self.df,
								'hist_cols':self.hist_cols,
								'cols':self.cols,
								'login':None,
								}
		else:
			self.df = st.session_state['dataframe']
			self.hist_cols = st.session_state['hist_cols']
			self.cols = st.session_state['cols']
		#
		self.__merge_tournament_and_deck_tables()

	def get_tournament_data(self):
		"""
		Method for returning all big tournament results in torunament table
		"""
		return self.tdf

	def get(self):
		'''
		Method for returng the processed data from the databse to
		the dashbord.
		'''
		return self.df, self.cols, self.hist_cols

	def __merge_tournament_and_deck_tables(self, ):
		"""
		Method for combining torunament dat to deck data for local results
		"""
		self.__merge_tournament_and_decks('Top', 'Local Top')
		self.__merge_tournament_and_decks('Teilnahme', 'Local')
		self.__merge_tournament_and_decks('Win', 'Local Win')

	def __merge_tournament_and_decks(self, result, new_name):
		"""
		Method for merging tournament data to deck data
		:param result: tournament satnding
		:param new_name: name of the tournament standing column
		:return result:merged dataframe
		"""
		# get local tops
		filtered = self.tdf[(self.tdf['Standing'] == result) & (self.tdf['Mode'] == 'Local')]
		result = (filtered[['Deck', 'Standing']].groupby('Deck').count())
		# merge
		self.df = pd.merge(
			self.df, result, left_on='Deck', right_on='Deck', how='left'
		).fillna(0)
		self.df.rename(columns={'Standing': new_name}, inplace=True)

	def __create_data_model(self):
		'''
		Method for getting and preprocessing the yugioh data from the
		database		
		:return df: dataframe with data
		:return cols: list with sorted columns of the dataframe
		'''
		# get data from data base
		df, hist_cols, save_cols = self.__get_all_data()
		hist_cols = self.__sort_hist_cols(hist_cols)
		df = self.__clean_data(df, hist_cols)
		return df, save_cols, hist_cols

	def __get_all_data(self, ):
		'''
		Method to get all data from the Database
		:use db: as detabase keys
		:return df_all: Dtaframe with all data like deck name, player and elo
		:return hist_cols: list of columns with historic elo data
		:return save_cols:list with all columns with further infos
		'''
		# get all data
		res = self.db.fetch()
		# create a dataframe
		df = pd.DataFrame(res.items)
		#df.to_csv('db.csv')
		# build a dataframe from the history
		tmp = df.loc[:, 'History'].to_list()
		hist_df = pd.DataFrame(tmp).fillna(0).astype(int)
		hist_cols = list(np.flip(hist_df.columns.to_list()))
		del df['History']
		save_cols = df.columns.to_numpy()
		save_cols = np.delete(save_cols, np.where(save_cols == 'key'))
		save_cols = np.delete(save_cols, np.where(save_cols == 'Meisterschaft'))
		# merge history dataframe and rest
		df_all = pd.merge(df, hist_df, right_index=True, left_index=True)
		
		return df_all, hist_cols, save_cols
	      
	def update_elo_ratings(self, deck1, deck2, erg1, erg2, df_elo, hist_cols, save_cols):
		'''
		Method for updating Elo Rating and result stats after a new match
		'''
		# get index
		idx1 = int(df_elo[df_elo['Deck']==deck1].index.to_numpy())
		idx2 = int(df_elo[df_elo['Deck']==deck2].index.to_numpy())
		# update siege/niederlagen
		if erg1 != erg2:
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

		df_elo = self.__update_tiers(df_elo)

		key1 = df_elo.at[idx1, 'key']
		key2 = df_elo.at[idx2, 'key']

		self.__update_element(key1, df_elo.loc[idx1, :], save_cols, hist_cols)
		self.__update_element(key2, df_elo.loc[idx2, :], save_cols, hist_cols)

	def __update_tiers(self, df):
		'''
		Method for updating tier standing of a deck after a match
		'''
		Label = ['Kartenstapel','Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0']
		NCl = len(Label)
		
		mean = int(df['Elo'].mean())
		std = float(df['Elo'].std())
		bins = np.array([0, mean-2*std, mean-std, mean, mean+std, mean+2*std, 10e6])
		liste =[]
		for jj in range(len(df)):    
			for kk in range(NCl):
				if(df.at[jj, 'Elo']>=bins[kk] and df.at[jj, 'Elo']<=bins[kk+1]):
					liste.append(Label[kk])
					break
		
		df['Tier'] = liste
		return df

	def __update_element(self, unique_key, ds, save_cols, hist_cols):
		'''
		Method for updating an element in the database
		:param unique_key: key of the element to updated
		:param ds: data series with the data with sould be updated
		:param save_cols: list of columns with element infos
		:param hist_cols: list with column names to build elo history
		:param db: key to database
		:return: nothing
		'''
		dict_update = {col: str(ds[col]) for col in save_cols}
		dict_hist = {col: str(ds[col]) for col in hist_cols}
		# include historic elo dictonary to info dictionary
		dict_update['History'] = dict_hist
		dict_update['Meisterschaft'] = ds['Meisterschaft']
		# update database
		self.db.put(dict_update, unique_key)
	      
	      
	def insert_tournament(self, result_dict):
		'''
		Method for updating the tournament column of a spezific deck
		:param result_dict: diconary with tournament results
		'''
		self.tourn_db.put(result_dict)
		
	      
	def update_stats(self, deck_choose, in_stats, modif_in, in_stats_type, new_type, df_elo, save_cols, hist_cols):
		'''
		Method for updating deck characteristica
		:param deck_choosen: name of the deck to update
		:param in_stats: info column to update
		:parm modif_in: value of the midification
		:param in_stats_type: update type of tthe choosen deck
		:param new_type: new_type to update
		:param df_elo: dataframe with all data
		:param save_cols: list with info columns
		:param hist_cols: columns with historic elo data
		:param db: key to database 
		:return: nothing
		'''
		# find index of choosen deck to update
		idx = int(df_elo[df_elo['Deck']==deck_choose].index.to_numpy())
		# update characteristics if choosen
		if len(in_stats) > 0:
			df_elo.at[idx, in_stats] = modif_in
		# update deck type if choosen
		if len(in_stats_type)>0:
			df_elo.at[idx, in_stats_type] = new_type
		# get deck key and update the database
		unique_key = df_elo.at[idx, 'key']
		df_elo['Meisterschaft'] = df_elo['Meisterschaft'].astype(object) 
		self.__update_element(unique_key, df_elo.loc[idx, :], save_cols, hist_cols)

	def update_history(self, df, hist_cols, save_cols):
		'''
		Method for increase the historic elo columns with the actual elo
		:param df: dataframe with all data
		:param hist_cols: list with historic elo columns
		:param save_cols: list with deck info columns
		:param db: key to database
		:return: nothing
		'''
		# get actual date and modify layout
		date = datetime.date.today()
		name = date.strftime("%m/%Y")
		# if actual date is not in historic columns include new column
		if name not in hist_cols:
			# include new column
			df[name] = df['Elo']
		# add new column to hist columns
		hist_cols += [name]
		# update all decks in database with new historic deck column
		for i in range(len(df)):
				unique_key = df.at[i, 'key']
				self.__update_element(unique_key, df.loc[i, :], save_cols, hist_cols)
		          
	def insert_new_deck(self, deck, name, atk, contr, rec, cons, combo, resi, typ):
		'''
		Function for inserting a new deck to the database
		:param deck: name of the deck
		:param name: name of the player
		:param atk: attack rating 
		:param contr: control rating
		:param rec: recovery rating
		:param cons: consistency rating
		:param combo: combo rating
		:param resi: resilience rating
		:param typ; type of the deck
		:param db: key to database
		:return: nothing
		'''
		# build info dictionary
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
		'Meisterschaft':self.df_default_meisterschaft,
		'Liga Pokal':0,	
		'Fun Pokal':0, 	
		'Matches':0,	
		'Siege':0,	
		'Remis':0,	
		'Niederlage':0,	
		'dgp':0, 
		'Elo':1500,
		'History':{}}
		# inserting the new deck
		self.db.put(input_dict)
	     
	def __clean_data(self, df, hist_cols):
		'''
		Method for cleaning and calculation of new data with data from the database
		:param df: dataframe with data from the database
		:param hist_cols: list with all historic elo columns
		:return df: dataframe with data in right format and new calculated values
		'''
		# columns with integer values
		cols_to_int = ['Elo', 'dgp', 'Wanderpokal', 'Liga Pokal', 'Fun Pokal', 'Attack', 'Combo', 'Resilience', 'Recovery',
					'Consistensy', 'Control','Matches', 'Siege', 'Remis', 'Niederlage']+hist_cols
		# fill nulls and transform float data to interes
		for k in cols_to_int:
			df[k] = df[k].fillna(0).astype(float)
			df[k] = df[k].astype(int)
		# last 3 month
		df = self.__calcualte_past_stats(df, 'Letzte 3 Monate', hist_cols, -1)
		# last 6 month
		df = self.__calcualte_past_stats(df, 'Letzte 6 Monate', hist_cols, -2)
		# last year
		df = self.__calcualte_past_stats(df, 'Letzte 12 Monate', hist_cols, -4)
		# calculate mean and std of the last year
		df['Mean 1 Year'] = np.nanmean(df[hist_cols[-5:-1]+['Elo']],1)
		df['Mean 1 Year'] = df['Mean 1 Year'].astype(int)
		df['Std. 1 Year'] = np.nanstd(df[hist_cols[-5:-1]+['Elo']],1)
		df['Std. 1 Year'] = df['Std. 1 Year'].astype(int)
		# calculate win rate
		df['Gewinnrate'] = 100*np.round(df['Siege']/(df['Siege']+df['Remis']+df['Niederlage']), 2)
		return df.fillna(0)

	def __calcualte_past_stats(self, df, new_column, hist_cols, past_idx):
		'''
		Method for calcualting the history elo stats changes of the decks
		:param df: deck datatframe
		:param new_column: name of the new column
		:param hist_cols: history columns
		:param past_idx: index of the past date for change calculation
		:return df: deck dataframe with new featuer
		'''
		df[new_column] = 0
		result = df[df[hist_cols[past_idx]] > 0].index
		df.loc[result, new_column] = df.loc[result, 'Elo'] - df.loc[result, hist_cols[past_idx]]
		df[new_column] = df[new_column].astype(int)
		return df

	def __sort_hist_cols(self, hist_cols):
		'''
		Method for sorting the list with historic elo in ascending order
		:param hist_cols: list with columns with historic elo 
		:return out: sorted list with historic elo
		'''
		t = [
			datetime.date(year=int(c[-4:]), month=int(c[:2]), day=1)
			for c in hist_cols
		]
		# sort dates
		t = pd.Series(t).sort_values().reset_index(drop=True)
		return [t[i].strftime('%m/%Y') for i in range(len(t))]

