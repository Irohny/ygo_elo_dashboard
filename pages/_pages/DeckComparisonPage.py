import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pages._pages.Visualization import Visualization

class DeckComparisionPage:
	'''
	Class for a page to compare the results and some statistics of 
	some choosen decks.
	'''
	def __init__(self, df, hist_cols):
		'''
			Init-Method
			:param df: dataframe with elo data
			:param hist_cols: column names of past elo stats
		'''
		# Set dashboard colors
		self.back_color = "#00172B"
		self.filed_color = "#0083B8"
		self.vis = Visualization()
		# build page
		self.__build_page_layout(df, hist_cols)

	def __build_page_layout(self, df, hist_cols):
		'''
		Method to build the page layout for elo history comparision
		:param df: dataframe with elo data
		:param hist_cols: column with names/dates of past elo stats
		:return: nothing
		'''
		# generate a Multiselect field for choosing the decks that elo stats should displayed
		deck = st.multiselect('Wähle deine Decks', options = list(df['Deck'].unique()), default = st.session_state['deck'])
		if not deck:
			st.stop()
		
		# define two tabs for an elo-plot and the statistic comparision
		plot, stats = st.tabs(['Eloverlauf', 'Statistiken'])
		# elo plot tab
		plot_cont = plot.container()
		# build a plotly figure with past elo data for choosen decks
		fig = self.__plot_elo_history(df, deck, hist_cols)
		plot_cont.plotly_chart(fig, use_container_width=True, theme=None, transparent=True)
		# statistic comparision
		# get number of rows for displaying deck stats
		# max 4 deck stats per row
		n_rows = int(np.ceil(len(deck)/4))
		counter = 0
		# loop over all rows
		for _ in range(n_rows):
			# two columns per deck stats
			columns = stats.columns([1,1,1,1,.001])
			for j in range(4):
				with columns[j]:
					if counter+j >= len(deck):
						continue
					idx = int(df[df['Deck']==deck[counter+j]].index.to_numpy())

					# header
					st.header(df.at[idx, 'Deck'])
					st.header(df.at[idx, 'Tier'])
					# image


			for i in range(4):
				if counter >= len(deck):
						break
				# get index and game result of current deck
				idx = int(df[df['Deck']==deck[counter]].index.to_numpy())
				values = df.loc[idx, ['Siege', 'Remis', 'Niederlage']].to_list()
				# display text statistics in even columns

				inside_col = columns[i].container(border=True).columns([1,1,1])
				lower_col = columns[i].container(border=True).columns([1,1,1])
				inside_col[0].metric(
					"Matches",
					f"{int(df.at[idx, 'Matches'])}",
					f"{int(np.sum(values))} Spiele",
				)
				#wanderpokal
				wanderpokal = ''.join(
					':trophy: '
					for _ in range(int(df.at[idx, 'Meisterschaft']['Win']['Wanderpokal']))
				)
				lower_col[0].markdown('Wanderpokal')
				lower_col[0].markdown(wanderpokal)
				# local wins
				localwin = ''.join(':medal: ' for _ in range(int(df.at[idx, 'Local Win'])))
				lower_col[0].markdown('Local Win')
				lower_col[0].markdown(localwin)
				#
				inside_col[1].metric('Aktuelle Elo', int(df.at[idx, 'Elo']), int(df.at[idx, 'Letzte 3 Monate']))
				funpokal = ''.join(
					':star: '
					for _ in range(int(df.at[idx, 'Meisterschaft']['Win']['Fun Pokal']))
				)
				lower_col[1].markdown('Fun Pokal')
				lower_col[1].markdown(funpokal)

				inside_col[2].metric('Gegner Stärke', int(df.at[idx, 'dgp']))
				localtop = ''.join(':star: ' for _ in range(int(df.at[idx,'Local Top'])))
				inside_col[2].markdown(' ')
				lower_col[2].markdown('Local Top')
				lower_col[2].markdown(localtop)

				# generate win/lose plot
				fig = self.vis.plotly_gauge_plot(100*df.at[idx,'Siege']/sum(values)//1)
				columns[i].plotly_chart(fig, transparent=True,  
										theme=None, use_container_width=True)

				categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
				fig = self.vis.make_spider_plot(df.loc[idx, categories].astype(int).to_list())
				columns[i].plotly_chart(fig, theme=None, transparent=True, use_container_width=True)
				# update counter for next deck of the decks of interest 
				counter += 1

	def __plot_elo_history(self, df, deck, hist_cols):
		'''
		Method for plotting the elo history of a set of choosen deck
		:param df: dataframe with deck elo ratings
		:param deck: list with choosen decks
		:param hist_cols: list with historic elo columns
		:return fig: plotly figure object
		'''
		# proof if a deck was choosen
		if len(deck)==0:
			return []
		# generate a date list
		dates = hist_cols+['Elo']
		df_plot = []
		# loop over all choosen decks and get elo of the dates
		for d in deck:
			tmp = pd.DataFrame(columns=['Deck', 'Elo', 'Date', 'helper'], index=range(len(dates)))
			tmp['Date'] = dates
			tmp['helper'] = np.linspace(0, len(dates)-1, len(dates))
			tmp['Deck'] = d
			tmp['Elo'] = df[df['Deck']==d][dates].to_numpy().squeeze()
			df_plot.append(tmp)
		# combine deck dataframe to a big dataframe
		df_plot = pd.concat(df_plot, axis=0)
		# set empty historic elo (value:0) to NaN and drop them
		df_plot[df_plot['Elo']==0]=np.nan
		#df_plot = df_plot.dropna()
		elo_min = df_plot['Elo'].min()-15
		elo_max = df_plot['Elo'].max()+15
		df_plot['Elo'] = df_plot['Elo'].fillna(0)
		df_plot = df_plot.sort_values(by='helper', ascending=True).reset_index(drop=True)
		# generate a plotly figure with elo ratings
		fig = px.line(df_plot, x="Date", y="Elo", color='Deck', line_shape='spline')
		# update figure traces for hover effects and line width
		fig.update_traces(textposition="bottom right", line=dict(width=5), hovertemplate=None)
		# update figure layout
		fig.update_layout(font_size=15, title='Eloverlauf', xaxis_title='Datum', yaxis_title='Elo',
						hovermode="x unified",yaxis=dict(range=[elo_min, elo_max]))
		
		return fig