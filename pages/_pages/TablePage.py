import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid

class TablePage:
	'''
	Class for a Tablepage for a look at all deck elos and the actual 	 ranking. MEthod includes some filter for better search options and
	overview
	'''
	def __init__(self, df):
		self.__build_page_layout(df)

	def __build_page_layout(self, df):
		'''
		'''
		left, right = st.columns([2, 4])
		with left:
				with st.form('filter'):
					slider_range = st.slider('Stellle den Elo-Bereich ein:', 
							min_value = int(df['Elo'].min()/10)*10,
							max_value = int(np.ceil(df['Elo'].max()/10))*10,
							value = [int(df['Elo'].min()/10)*10, int(np.ceil(df['Elo'].max()/10))*10],
							step = 10)
					# filter for players
					owner = st.multiselect(
							"Wähle Spieler:",
							options = df['Owner'].unique(),
							default = st.session_state['owner'])
					# filter for showing the data
					types = st.multiselect(
							'Wähle Decktyp',
							options = df['Type'].unique(),
							default = st.session_state['types_select'])
					
					# filter for gruppe
					tier = st.multiselect(
							"Wähle die Deckstärke:",
							options = df['Tier'].unique(),
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
					idx.append(df[df[tour]>0].index.to_numpy())
				idx = np.concatenate(idx)
				df_select = df.loc[idx, :].reset_index(drop=True)
		else:
				df_select = df.copy()
					
		df_select = df_select.query(
				"Owner == @owner & Tier == @tier & Type == @types"
				)
		df_select = df_select[(df_select['Elo']>=slider_range[0]) & (df_select['Elo']<=slider_range[1])]
		df_select = df_select.reset_index()
		
		with right:
				df_select = df_select[['Deck', 'Elo', 'Gewinnrate', 'Letzte 3 Monate', 'Letzte 6 Monate', 'Letzte 12 Monate', 'Matches', 'Siege', 'Remis', 'Niederlage']].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
				df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']] = df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']].astype(int)
				df_select['Platz'] = df_select.index.to_numpy()+1
				AgGrid(df_select[['Platz', 'Deck', 'Elo', 'Gewinnrate', 'Letzte 3 Monate', 'Letzte 6 Monate', 'Letzte 12 Monate', 'Matches', 'Siege', 'Remis', 'Niederlage']], fit_columns_on_grid_load=True, update_mode='NO_UPDATE',
				theme='streamlit', )