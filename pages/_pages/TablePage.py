import numpy as np
import pandas as pd
import streamlit as st

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
		expander = st.expander(':mag: Filter', expanded=False)
		form = expander.form('table_filter')
		cols = form.columns([1,1,1,1])
		# slider for elo range
		slider_range = cols[0].slider('Stellle den Elo-Bereich ein:', 
				min_value = int(df['Elo'].min()/10)*10,
				max_value = int(np.ceil(df['Elo'].max()/10))*10,
				value = [int(df['Elo'].min()/10)*10, int(np.ceil(df['Elo'].max()/10))*10],
				step = 10)
		
		# filter for players
		player_list = list(df['Owner'].unique())
		df_player = pd.DataFrame(index=player_list, columns=['Wähle Spieler'])
		df_player['Wähle Spieler'] = True
		owner = cols[1].data_editor(df_player)
		owner = owner[owner['Wähle Spieler']].index.to_list()

		# filter for showing the data
		list_types = list(df['Type'].unique())
		df_type = pd.DataFrame(index=list_types, columns=['Wähle Deckstrategien'])
		df_type['Wähle Deckstrategien'] = True
		types = cols[2].data_editor(df_type)
		types = types[types['Wähle Deckstrategien']].index.to_list()
		
		# filter for gruppe
		df_tier = pd.DataFrame([True, True, True, True, True, True],
						 index=['Tier 0', 'Tier 1', 'Tier 2', 'Good', 'Fun', 'Kartenstapel'],columns=['Wähle die Deckstärke'])
		tier = cols[3].data_editor(df_tier)
		tier = tier[tier['Wähle die Deckstärke']].index.to_list()
		# filter for modus
		st.session_state['tournament'] = cols[0].selectbox(
				'Suche nach einem Turnier:',
				options = ['Alle', 'Wanderpokal', 'Lokal Teilnahme', 'Lokal Top', 'Lokal Win'])
		
		form.form_submit_button('Aktualisiere Tabelle') 
		#	          
		if st.session_state['tournament'] == 'Alle':
				df_select = df.copy()
		else:
				idx = []
				for idx_deck in df.index.to_list():
					if st.session_state['tournament'] == 'Wanderpokal' and df.at[idx_deck, 'Meisterschaft']['Win']['Wanderpokal']>0:
						idx.append(idx_deck)
					elif st.session_state['tournament'] == 'Lokal Teilnahme' and df.at[idx_deck, 'Meisterschaft']['Teilnahme']['Local']>0:
						idx.append(idx_deck)
					elif st.session_state['tournament'] == 'Lokal Top' and df.at[idx_deck, 'Meisterschaft']['Top']['Local']>0:
						idx.append(idx_deck)
					elif st.session_state['tournament'] == 'Lokal Win' and df.at[idx_deck, 'Meisterschaft']['Win']['Local']>0:
						idx.append(idx_deck)
				if idx:
					idx = np.concatenate(idx)
					df_select = df.loc[idx, :].reset_index(drop=True)
				else:
					df_select = df.copy()
					st.error('Keine Decks gefunden')
					
		df_select = df_select.query(
				"Owner == @owner & Tier == @tier & Type == @types"
				)
		df_select = df_select[(df_select['Elo']>=slider_range[0]) & (df_select['Elo']<=slider_range[1])]
		df_select = df_select.reset_index()
		df_select = df_select[['Deck', 'Elo', 'Gewinnrate', 'Letzte 3 Monate', 'Letzte 6 Monate', 'Letzte 12 Monate', 'Matches', 'Siege', 'Remis', 'Niederlage']].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
		df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']] = df_select[['Elo', 'Matches', 'Siege', 'Remis', 'Niederlage']].astype(int)
		df_select['Platz'] = df_select.index.to_numpy()+1
		df_select['Gewinnrate'] = np.round(df_select['Gewinnrate'],2)
		st.dataframe(df_select[['Platz','Deck', 'Elo', 'Gewinnrate', 'Letzte 3 Monate', 'Letzte 6 Monate', 'Letzte 12 Monate', 'Matches', 'Siege', 'Remis', 'Niederlage']], 
					height=820,hide_index=True, use_container_width=True)
		
