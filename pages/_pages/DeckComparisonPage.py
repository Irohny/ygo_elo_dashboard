import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
		# define two tabs for an elo-plot and the statistic comparision
		plot, stats = st.tabs(['Eloverlauf', 'Statistiken'])
		# elo plot tab
		with plot:
			with st.container():
				# build a plotly figure with past elo data for choosen decks
				fig = self.__plot_elo_history(df, deck, hist_cols)
				if len(deck)>0:
					st.plotly_chart(fig, theme=None, use_container_width=True)
		# statistic comparision 
		with stats:   
			if len(deck) > 0:
				# gt number of rows for displaying deck stats
				# max 4 deck stats per row
				n_rows = int(np.ceil(len(deck)/4))
				counter = 0
				# loop over all rows
				for r in range(n_rows):
						# define columns for stats displaying
						col_names = st.columns(4)
						for j in range(4):
							with col_names[j]:
								if counter+j >= len(deck):
									continue
								idx = int(df[df['Deck']==deck[counter+j]].index.to_numpy())
								st.header(df.at[idx, 'Deck'])
						# two columns per deck stats
						columns = st.columns([1.8,3.2,1.8,3.2,1.8,3.2,1.8,3.2])
						for i in range(4):
							if counter >= len(deck):
									break
							# get index and game result of current deck
							idx = int(df[df['Deck']==deck[counter]].index.to_numpy())
							values = df.loc[idx, ['Siege', 'Remis', 'Niederlage']].to_list()
							# display text statistics in even columns
							with columns[2*i]:
									st.subheader(df.at[idx, 'Tier'])
									st.caption('Wanderpokal:   ' + ':star:'*int(df.at[idx, 'Meisterschaft']['Win']['Wanderpokal']))
									st.caption('Fun Pokal:     ' + ':star:'*int(df.at[idx, 'Meisterschaft']['Win']['Fun Pokal']))
									st.caption('Lokal Win:   ' + ':star:'*int(df.at[idx, 'Meisterschaft']['Win']['Local']))
									st.caption('Lokal Top:     ' + ':star:'*int(df.at[idx, 'Meisterschaft']['Top']['Local']))
									st.metric(f"Matches", f"{int(df.at[idx, 'Matches'])}", f"{int(np.sum(values))} Spiele")
									st.metric('Aktuelle Elo', int(df.at[idx, 'Elo']), int(df.at[idx, 'Letzte 3 Monate']))
									st.metric('Gegner Stärke', int(df.at[idx, 'dgp']))
									
							# displaying win/lose plot and stats spider in odd columns		
							with columns[2*i+1]:
									st.header(" ")
									st.header(" ")
									# generate win/lose plot
									fig = self.__semi_circle_plot(values)
									st.pyplot(fig, transparent=True)
									# generate stats spider plot
									categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
									fig = self.__make_spider_plot(df.loc[idx, categories].astype(int).to_list())
									st.plotly_chart(fig, theme="streamlit", use_container_width=True)
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
		min = df_plot['Elo'].min()-15
		max = df_plot['Elo'].max()+15
		df_plot['Elo'] = df_plot['Elo'].fillna(0)
		df_plot = df_plot.sort_values(by='helper', ascending=True).reset_index(drop=True)
		# generate a plotly figure with elo ratings
		fig = px.line(df_plot, x="Date", y="Elo", color='Deck')
		# update figure traces for hover effects and line width
		fig.update_traces(textposition="bottom right", line=dict(width=5), hovertemplate=None)
		# update figure layout
		fig.update_layout(font_size=15, template='plotly_white', title='Eloverlauf', xaxis_title='Datum', yaxis_title='Elo',
							paper_bgcolor=self.back_color, plot_bgcolor=self.back_color, hovermode="x unified",yaxis=dict(range=[min,max]))
		
		return fig
	
	def __make_spider_plot(self, stats):
		'''
		Method to mmake a spider plot to show deck stats
		:param stats: list with stats of the current deck
		return fig: plotly figure object
		'''
		categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience', 'Attack']
		# append first item again for closen line figure in spider plot
		stats.append(stats[0])
		# generate plotly figure object and set layout
		plot = go.Scatterpolar(r=stats, theta=categories, fill='toself', line=dict(color=self.filed_color, width=3))
		fig = go.Figure(data = [plot], layout=go.Layout(paper_bgcolor=self.back_color, plot_bgcolor=self.back_color))
		fig.update_layout(font_size=15, 
		    			polar=dict(radialaxis=dict(visible=False, range=[0, 5])),
  						showlegend=False,
						hovermode="x",
						title='Deck Eigenschaften'
						)
		return fig

	def __semi_circle_plot(self, val):
		'''
		Method for plotting a semi circle plot to show number of wins and losses
		:param val: list with win/remis/loss standing
		:return fig: maplotlib figure object
		'''
		# generate bottom part of the dount plot
		# set lower part to background color for getting a semi circle plot
		val = np.append(val, sum(val))  # 50% blank
		colors = ['green', 'blue', 'red', self.back_color]
		explode= 0.05*np.ones(len(val))
		# plot with data and layouting
		fig = plt.figure(figsize=(8,4))
		ax = fig.add_subplot(1,1,1)
		ax.pie(val, colors=colors, pctdistance=0.85, explode=explode)
		ax.text(-1.05, 1.1, f"N {int(val[2]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=15)
		ax.text(-0.1, 1.1, f"U {int(val[1]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=15)
		ax.text(0.65, 1.1, f"S {int(val[0]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=15)
		ax.add_artist(plt.Circle((0, 0), 0.6, color=self.back_color))
		
		return fig