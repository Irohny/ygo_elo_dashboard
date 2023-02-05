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
		self.back_color = "#00172B"
		self.filed_color = "#0083B8"
		self.__build_page_layout(df, hist_cols)

	def __build_page_layout(self, df, hist_cols):
		'''
		Method to build the page layout for elo history comparision
		:param df:
		:param hist_cols:
		:return: nothing
		'''
		deck = st.multiselect('Wähle deine Decks', options = list(df['Deck'].unique()), default = st.session_state['deck'])
		plot, stats = st.tabs(['Eloverlauf', 'Statistiken'])
		with plot:
			with st.container():
				fig = self.__plot_elo_history(df, deck, hist_cols)
				if len(deck)>0:
					st.plotly_chart(fig, theme=None, use_container_width=True) 
		with stats:   
			st.markdown('# Überblick')
			if len(deck) > 0:
				n_rows = int(np.ceil(len(deck)/4))
				counter = 0
				for r in range(n_rows):
						columns = st.columns([1.8,3.2,1.8,3.2,1.8,3.2,1.8,3.2])
						for i in range(4):
							if counter >= len(deck):
									break
							
							idx = int(df[df['Deck']==deck[counter]].index.to_numpy())
							values = df.loc[idx, ['Siege', 'Remis', 'Niederlage']].to_list()
							
							with columns[2*i]:
									st.header(df.at[idx, 'Deck']+" :star:"*int(df.at[idx, 'Wanderpokal']))
									st.subheader(df.at[idx, 'Tier'])
									st.subheader(f"Matches: {int(df.at[idx, 'Matches'])}")
									st.subheader(f"Spiele: {int(np.sum(values))}")
									st.metric('Aktuelle Elo', int(df.at[idx, 'Elo']), int(df.at[idx, 'Letzte 3 Monate']))
									st.metric('Gegner Stärke', int(df.at[idx, 'dgp']))
									st.caption('Wanderpokal:   ' + ':star:'*int(df.at[idx, 'Wanderpokal']))
									st.caption('Fun Pokal:     ' + ':star:'*int(df.at[idx, 'Fun Pokal']))
									st.caption('Meisterschaft: ' + ':star:'*int(df.at[idx, 'Meisterschaft']))
									st.caption('Liga Pokal:    ' + ':star:'*int(df.at[idx, 'Liga Pokal']))
									
							with columns[2*i+1]:
									st.header(" ")
									st.header(" ")
									
									fig = self.__semi_circle_plot(values)
									st.pyplot(fig, transparent=True)
									
									categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
									fig = self.__make_spider_plot(df.loc[idx, categories].astype(int).to_list())
									st.plotly_chart(fig, theme="streamlit", use_container_width=True)
									
									counter += 1

	def __plot_elo_history(self, df, deck, hist_cols):
		'''
		Method for plotting the elo history of a set of choosen deck
		:param df: dataframe with deck elo ratings
		:param deck: list with choosen decks
		:param hist_cols: list with historic elo columns
		:return fig: plotly figure object
		'''
		if len(deck)==0:
			return []
		dates = hist_cols+['Elo']
		df_plot = []
		for d in deck:
			tmp = pd.DataFrame(columns=['Deck', 'Elo', 'Date'])
			tmp['Date'] = dates
			tmp['Deck'] = d
			tmp['Elo'] = df[df['Deck']==d][dates].to_numpy().squeeze()
			df_plot.append(tmp)
		df_plot = pd.concat(df_plot, axis=0)
		df_plot[df_plot['Elo']==0]=np.nan
		df_plot = df_plot.dropna().reset_index(drop=True)
		fig = px.line(df_plot, x="Date", y="Elo", color='Deck')
		fig.update_traces(textposition="bottom right", line=dict(width=5), hovertemplate=None)#
		fig.update_layout(font_size=15, template='plotly_white', title='Eloverlauf', xaxis_title='Datum', yaxis_title='Elo',
							paper_bgcolor=self.back_color, plot_bgcolor=self.back_color, hovermode="x unified")
		
		return fig
	
	def __make_spider_plot(self, stats):
		'''
		Method to mmake a spider plot to show deck stats
		:param stats: list with stats of the current deck
		return fig: plotly figure object
		'''
		categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience', 'Attack']
		#df_plot = pd.DataFrame(categories, columns=['Kategorie'])
		stats.append(stats[0])
		plot = go.Scatterpolar(r=stats, theta=categories, fill='toself', line=dict(color=self.filed_color, width=3))
		fig = go.Figure(data = [plot], layout=go.Layout(paper_bgcolor=self.back_color, plot_bgcolor=self.back_color))
		#fig.update_polars(radialaxis=dict(]))
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
		:return fig:
		'''
		val = np.append(val, sum(val))  # 50% blank
		colors = ['green', 'blue', 'red', self.back_color]
		explode= 0.05*np.ones(len(val))
		# plot
		fig = plt.figure(figsize=(8,5))
		ax = fig.add_subplot(1,1,1)
		#p = patches.Rectangle((left, bottom), width, height,fill=False, transform=ax.transAxes, clip_on=False)

		ax.pie(val, colors=colors, pctdistance=0.85, explode=explode)
		ax.text(-1.05, 1.1, f"N {int(val[2]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
		ax.text(-0.1, 1.1, f"U {int(val[1]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
		ax.text(0.65, 1.1, f"S {int(val[0]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
		ax.add_artist(plt.Circle((0, 0), 0.6, color=self.back_color))
		
		return fig