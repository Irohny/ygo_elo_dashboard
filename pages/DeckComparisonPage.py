import numpy as np
import pandas as pd
import streamlit as st
import VisualizationTools as vit

class DeckComparisionPage:
	'''
	Class for a page to compare the results and some statistics of 
	some choosen decks.
	'''
	def __init__(self):
		'''
			Init-Method
			:param df: dataframe with elo data
			:param hist_cols: column names of past elo stats
		'''
		self.vis = vit.Visualization()
		self.decks_per_row = 5
		# build page
		self.df = st.session_state['deck_data'].copy()
		self.hist_cols = st.session_state['history_columns']
		self.tournament = st.session_state['tournament_data'].copy()
		self.__build_page_layout()

	def __build_page_layout(self):
		'''
		Method to build the page layout for elo history comparision
		:param df: dataframe with elo data
		:param hist_cols: column with names/dates of past elo stats
		:return: nothing
		'''
		st.title(':trophy: Deckvergleich :trophy:', anchor='anchor_tag')
		# generate a Multiselect field for choosing the decks that elo stats should displayed
		decks = st.multiselect('Wähle deine Decks', options = list(self.df['Deck'].unique()), 
							default = st.session_state['deck'])
		if not decks:
			st.stop()
		df_decks = self.df[self.df['Deck'].isin(decks)].reset_index(drop=True)

		# build a plotly figure with past elo data for choosen decks
		fig = self.vis.deck_lineplot(df_decks, self.hist_cols)
		st.plotly_chart(fig, use_container_width=True, theme=None, transparent=True)
		
		# statistic comparision
		n_rows = int(np.ceil(len(decks)/self.decks_per_row))
		
		# loop over all rows
		for n_row in range(n_rows):
			# two columns per deck stats
			columns = st.columns(self.decks_per_row)
			for j in range(self.decks_per_row):
				idx_deck = int(n_row*self.decks_per_row+j)
				if idx_deck >= len(decks):
					continue
				# get deck data and combine data
				deck = decks[idx_deck]
				df_tour = self.tournament[self.tournament['Deck']==deck].reset_index(drop=True)
				deck = df_decks[df_decks['Deck']==deck].iloc[0]
				values = [deck['Siege'], deck['Remis'], deck['Niederlage']]
				if df_tour.empty:
					tour_values = [0,0,0]
					local_wins, local_tops = 0, 0
				else:
					tour_values = [df_tour['Win'].sum(), df_tour['Draw'].sum(), df_tour['Loss'].sum()]
					local_tops = sum((df_tour['Mode']=='Local')&(df_tour['Standing']=='Top'))
					local_wins = sum((df_tour['Mode']=='Local')&(df_tour['Standing']=='Win'))
					
				act_col = columns[j].container(border=True)
				act_col.header(deck['Deck'])
				img_col = act_col.columns(2)
				img_col[0].subheader(deck['Tier'])
				img_col[0].subheader(deck['Type'])
				img_col[0].metric('Aktuelle Elo', int(deck['Elo']), int(deck['Letzte 3 Monate']))
				#
				vit.load_and_display_image(f"./Deck_Icons/{deck['Deck']}.png",
											pos=img_col[1])
				
				# get index and game result of current deck
				# display text statistics in even columns

				inside_col = act_col.columns([1,1,1])
				inside_col[0].metric( "Matches", f"{int(deck['Matches'])}",f"{int(sum(values))} Spiele")
				inside_col[1].metric('Gegner Stärke', int(deck['dgp']))
				inside_col[2].metric('Turniere', len(df_tour), f"{tour_values[0]}/{tour_values[1]}/{tour_values[2]}")
				act_col.markdown('---')
				# Pokale
				inside_col = act_col.columns([1,1,1])
				wanderpokal = ''.join(':trophy: ' for _ in range(int(deck['Meisterschaft']['Win']['Wanderpokal'])))
				inside_col[0].markdown('Wanderpokal')
				inside_col[0].markdown(wanderpokal)
				# local wins
				localwin = ''.join(':medal: ' for _ in range(local_wins))
				inside_col[0].markdown('Local Win')
				inside_col[0].markdown(localwin)
				#
				funpokal = ''.join(
					':star: '
					for _ in range(int(deck['Meisterschaft']['Win']['Fun Pokal']))
				)
				inside_col[1].markdown('Fun Pokal')
				inside_col[1].markdown(funpokal)

				localtop = ''.join(':star: ' for _ in range(local_tops))
				inside_col[2].markdown('Local Top')
				inside_col[2].markdown(localtop)

				# generate win/lose plot
				tabs = act_col.tabs(['Gewinrate', 'Eigenschaften', 'Turniere'])
				# gewinrate
				fig = self.vis.plotly_gauge_plot(100*values[0]/sum(values)//1)
				tabs[0].plotly_chart(fig, transparent=True, use_container_width=True)
				# typen
				categories = [int(deck['Attack']), int(deck['Control']), int(deck['Recovery']), 
				  			int(deck['Consistensy']), int(deck['Combo']), int(deck['Resilience'])]
				fig = self.vis.make_spider_plot(categories)
				tabs[1].plotly_chart(fig, transparent=True, use_container_width=True)
				# turnier rate
				if df_tour.empty:
					tabs[2].error('Keine Turnierergebnisse.')
				else:
					fig = self.vis.plotly_gauge_plot(100*tour_values[0]/sum(tour_values)//1)
					tabs[2].plotly_chart(fig, transparent=True, use_container_width=True)
					
				
				