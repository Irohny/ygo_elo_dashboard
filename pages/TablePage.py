import numpy as np
import pandas as pd
import streamlit as st
from VisualizationTools import Visualization
class TablePage:
	'''
	Class for a Tablepage for a look at all deck elos and the actual 	 ranking. MEthod includes some filter for better search options and
	overview
	'''
	def __init__(self):
		self.df = st.session_state['deck_data']
		self.tournament = st.session_state['tournament_data']
		self.visu = Visualization()
		self.__build_page_layout()
		
	def __build_page_layout(self):
		'''
		'''
		st.title(':trophy: Ewige Tabelle :trophy:', anchor='table')
		cols = st.columns([1,4])
		filter_area = cols[0].container(border=True)
		table_categorie = filter_area.radio('Wertung', ['Elo', 'Turnierscore'], horizontal=True)
		self.activ_decks = filter_area.toggle('Aktive Decks', value=False)
		self.__create_popover_filter(filter_area)
		
		df_select = self.__use_filter_for_table()
		# Setup selected data	          
		if table_categorie == 'Elo':
			# 
			data = df_select[['Platz','Deck', 'Elo', 'Gewinnrate', 'Letzte 3 Monate', 'Letzte 6 Monate', 'Letzte 12 Monate', 'Matches', 'Siege', 'Remis', 'Niederlage']]
			subset = ["Letzte 3 Monate", "Letzte 6 Monate","Letzte 12 Monate"]
			data[subset] = data[subset].astype(int)
			data['Gewinnrate'] = data['Gewinnrate'].astype(int)
			#data['Sieg | Remis | Niederlage'] = [[row['Siege'], row['Remis'], row['Niederlage']] for _, row in data[['Siege', 'Remis', 'Niederlage']].iterrows()]
			#data.drop(columns=['Siege', 'Remis', 'Niederlage'], inplace=True)
			data = data.style.format(self.__format_arrow, subset=subset).applymap(self.__color_arrow, subset=subset)
			
			config = {'Platz':st.column_config.NumberColumn(width='small'),
					'ELO':st.column_config.NumberColumn(width='small'),
					"Letzte 3 Monate":st.column_config.NumberColumn('Letzte Änderung', width='small'),
					'Gewinnrate':st.column_config.NumberColumn(width='small', format='%.2f')}
		else:
			
			data = df_select[['Platz', 'Deck', 'Tourn_Win', 'Tourn_Loss', 'Tourn_Draw', 'Turniere',
					   		'Top-Rate', 'Match-Win-Rate', 'Points', ]]
			data.dropna(inplace=True)
			data.sort_values('Points', ascending=False, ignore_index=True, inplace=True)
			data['Platz'] = data.index.to_numpy()+1
			config = {'Platz':st.column_config.NumberColumn('Platz', width='small'),
			 		'Deck':st.column_config.TextColumn('Deck', width='medium'), 
					'Tourn_Win':st.column_config.NumberColumn('Siege', width='small'), 
					'Tourn_Loss':st.column_config.NumberColumn('Niederlagen', width='small'), 
					'Tourn_Draw':st.column_config.NumberColumn('Remis', width='small'), 
					'Turniere':st.column_config.NumberColumn('Turniere', width='small'),   		
					'Top-Rate':st.column_config.NumberColumn('Top-Rate', width='small'), 
					'Match-Win-Rate':st.column_config.NumberColumn('Win-Rate', width='small'), 
					'Points':st.column_config.NumberColumn('Punkte', width='small')}
		cols[1].dataframe(data, 
					height=820,
					hide_index=True, 
					use_container_width=True,
					column_config=config)
		
		# Costum Tabel Component (not useable now :c)
		"""c = st.columns(len(data.columns), vertical_alignment='top')
		for i, row in data.iterrows():
			for f, feat in enumerate(data.columns):
				if i == 0:
					c[f].button(feat, use_container_width=True)
				color = 'white'
				if feat in {'Letzte 3 Monate', 'Letzte 6 Monate', 'Letzte 12 Monate'}:
					if float(row[feat])>0:
						color = 'green' 	
					elif float(row[feat])<0:
						color = 'red' 
				if feat == 'Sieg | Remis | Niederlage':
					fig = self.visu.stacked_bar(data.at[i, 'Sieg | Remis | Niederlage'])
					c[f].plotly_chart(fig, theme='streamlit', height=10) 
				else:
					c[f].markdown(f"<p style='text-align: center; color: {color};' >{row[feat]}</p>", unsafe_allow_html=True)
				#c[f].markdown('---')"""
		

	def __color_arrow(self, val):
		return "color: green" if val > 0 else "color: red" if val < 0 else "color: white"

	def __format_arrow(self, val):
		return f"{'↑' if val > 0 else '↓'} {abs(val):.0f}" if val != 0 else f"{val:.0f}"

	def __create_popover_filter(self, col=st):
		"""
		
		"""
		expander = col.popover(':mag: Filter')
		form = expander.form('table_filter')
		
		# 1. slider for elo range
		self.slider_range = form.slider('Stellle den Elo-Bereich ein:', 
				min_value = int(self.df['Elo'].min()/10)*10,
				max_value = int(np.ceil(self.df['Elo'].max()/10))*10,
				value = [int(self.df['Elo'].min()/10)*10, int(np.ceil(self.df['Elo'].max()/10))*10],
				step = 10)
		
		cols = form.columns([1,1,1])
		# 2. filter for players
		player_list = list(self.df['Owner'].unique())
		self.owner = self.__selection_checkbox(player_list, 'Spieler', cols[0])
		
		# 3. filter for types
		list_types = list(self.df['Type'].unique())
		self.types = self.__selection_checkbox(list_types, 'Decktypen', cols[1])
		
		# 4. filter for Tier					 
		self.tier = self.__selection_checkbox(['Tier 0', 'Tier 1', 'Tier 2', 'Good', 'Fun', 'Kartenstapel'], 
										'Deck-Tiers', cols[2])
		 
		# 5. filter for tournament
		#st.session_state['tournament'] = cols[0].selectbox(
		#		'Suche nach einem Turnier:',
		#		options = ['Alle', 'Wanderpokal', 'Local'])
		
		form.form_submit_button('Aktualisiere Tabelle')

	def __use_filter_for_table(self)->pd.DataFrame:
		
		"""if st.session_state['tournament'] == 'Alle':
				df_select = self.df.copy()
		else:
			idx = []
			for idx_deck in self.df.index.to_list():
				if st.session_state['tournament'] == 'Wanderpokal' and self.df.at[idx_deck, 'Meisterschaft']['Win']['Wanderpokal']>0:
					idx.append(idx_deck)
				elif st.session_state['tournament'] == 'Lokal Teilnahme' and self.df.at[idx_deck, 'Meisterschaft']['Teilnahme']['Local']>0:
					idx.append(idx_deck)
				elif st.session_state['tournament'] == 'Lokal Top' and self.df.at[idx_deck, 'Meisterschaft']['Top']['Local']>0:
					idx.append(idx_deck)
				elif st.session_state['tournament'] == 'Lokal Win' and self.df.at[idx_deck, 'Meisterschaft']['Win']['Local']>0:
					idx.append(idx_deck)
			if idx:
				idx = np.concatenate(idx)
				df_select = self.df.loc[idx, :].reset_index(drop=True)
			else:
				df_select = self.df.copy()
				st.error('Keine Decks gefunden')
		"""
		df_select = self.df.copy()
		if self.activ_decks:
			df_select = df_select[df_select['active']]
		# filter for owner tier and type
		df_select = df_select[df_select['Owner'].isin(self.owner)]
		df_select = df_select[df_select['Tier'].isin(self.tier)]
		df_select = df_select[df_select['Type'].isin(self.types)]
		df_select = df_select[(df_select['Elo']>=self.slider_range[0]) & (df_select['Elo']<=self.slider_range[1])]
		df_select = df_select.reset_index()
		
		df_select = df_select.sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
		#df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']] = df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']].astype(int)
		df_select['Platz'] = df_select.index.to_numpy()+1
		df_select['Gewinnrate'] = np.round(df_select['Gewinnrate'],2)
		return df_select
	
	def __selection_checkbox(self, list_data:list, title:str, col:st):
		col.caption(title)
		res = []
		list_data = np.array(list_data)
		for i, data in enumerate(list_data):
			res.append(col.checkbox(data, value=True))
		return list_data[res]