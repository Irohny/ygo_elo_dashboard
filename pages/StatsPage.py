import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
import VisualizationTools as vit

class StatsPage:
	'''
	Class for generating the firste page of the dashboard with
	some major statistics
	'''
	def __init__(self):
		'''
		Init-Method
		:param self.df: dataframe with elo data
		:param st.session_state['tournament_data']: dataframe with tournament data
		:param self.hist_cols: list of column names with old elo
		'''
		self.df = st.session_state['deck_data'].copy()
		self.hist_cols = st.session_state['history_columns']
		# build page layout
		self.axis_color = 'white'
		self.vis = vit.Visualization()
		self.__build_page_layout()

	def __build_page_layout(self):
		'''
		Method for layouting the the front page
		:param self.df: dataframe with elo data
		:param st.session_state['tournament_data']: dataframe with tournament data
		:param self.hist_cols: list with column names of old elo data 
		'''
		# get number of all decks
		n = len(self.df['Deck'].unique())
		st.title(':trophy: YuGiOh! Elo-Dashboard :trophy:', anchor='anchor_tag')
		self.__first_kpi_row(n)
		self.__second_kpi_row()
				
			
	def __first_kpi_row(self, n):
		'''
		Method for displaying the first row of the front page KPI's
		Show number of player, games, deck
		:param self.df: dataframe with elo/deck data
		:param st.session_state['tournament_data']: dataframe with tournament data
		:param n: number of all decks 
		'''
		cols = st.columns(3)
		# first box with number of decks and matches
		total_box = cols[0].container(border=True).columns(2, vertical_alignment='bottom')
		total_box[0].metric(":flower_playing_cards: Decks in Wertung:", n) 
		total_box[1].metric(":flower_playing_cards: Aktive Decks:", sum(self.df['active'])) 
		
		# second box with number of games and player
		player_box = cols[1].container(border=True).columns(3)
		player_box[0].metric(":crossed_swords: Gesamtzahl Matches:", int(self.df['Matches'].sum()/2))
		player_box[1].metric(":crossed_swords: Gesamtzahl Spiele:", int((self.df['Siege'].sum()+self.df['Remis'].sum()+self.df['Niederlage'].sum())/2))
		player_box[2].metric(":bust_in_silhouette: Anzahl Spieler:", len(self.df['Owner'].unique()))
		# third box with number of tournaments and top rating
		tournament_box = cols[2].container(border=True).columns(2)
		tournament = st.session_state['tournament_data'].copy()
		tournament_box[0].metric(':bank: Turnierteilnahmen', len(tournament))
		tournament_box[1].metric(':medal: Top-Rate', f'{np.round(len(tournament[tournament["Standing"]=="Top"])/len(tournament)*100,2)}%')

	def __second_kpi_row(self):
		'''
		Method for displaying the second KPI row of the dashboard frontpage
		:param self.df: dataframe with elo/deck data
		:param st.session_state['tournament_data']: dataframe with tournament data
		:param self.hist_cols: list of columns names with old elo
		'''
		# get best and worst decks in the data
		# define layout columns with header for the following icons and metrics
		cols = st.columns([1,1,1], vertical_alignment='top')
		self.__second_row_left(cols[0])
		self.__second_row_middle(cols[1])
		self.__second_row_right(cols[2])

	def __second_row_left(self, col0:st):
		# first column: all over best elo
		selection = col0.radio('ScoreControl', ['Turnierscore', 'Beste Elo', 'Schleteste Elo'],
						 horizontal=True, label_visibility='collapsed')
		if selection in ['Beste Elo', 'Schleteste Elo']:
			self.__deck_elo_widget(selection, col0)
		else:
			row = self.df[self.df['Points']==self.df['Points']].iloc[0]
			c = col0.columns(2)
			vit.load_and_display_image(pos=c[0],
									path=f'./Deck_Icons/{row["Deck"]}.png',
									subtext = f'{row["Deck"]} {row["Elo"]}')
			tmp = self.df[self.df['Platz']>0].copy().sort_values('Platz', ignore_index=True)
			tmp.rename(columns={'Points':'Punkte'}, inplace=True)
			c[1].dataframe(tmp[['Platz','Deck','Punkte']].head(10),hide_index=True)
			
		
	def __deck_elo_widget(self, header:str, col:st):
		"""
		"""
		if header == 'Beste Elo':
			df_max = self.df[self.hist_cols+['Elo']].agg(['idxmax', 'max'])
		
			deck_actuel = self.df.at[int(df_max.at['idxmax', 'Elo']), 'Deck']
			elo_actuel = int(df_max.at['max', 'Elo'])

			deck_ever = self.df.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']
			elo_ever = int(df_max.loc['max', :].max())
		elif header == 'Schleteste Elo':
			df_min = self.df[self.df[self.hist_cols+['Elo']]>0][self.hist_cols+['Elo']].agg(['idxmin', 'min'])
		
			deck_ever = self.df.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']
			elo_ever = int(df_min.loc['min', :].min())
		
			deck_actuel = self.df.at[df_min.loc['idxmin', 'Elo'], 'Deck']
			elo_actuel = int(df_min.loc['min', 'Elo'].min())
		c = col.columns(2)
		vit.load_and_display_image(pos=c[0],
									title='Aktuell',
									path=f'./Deck_Icons/{deck_actuel}.png',
									subtext = f"{deck_actuel} {elo_actuel}")
		vit.load_and_display_image(pos=c[1],
									title='Total',
									path=f'./Deck_Icons/{deck_ever}.png',
									subtext = f"{deck_ever} {elo_ever}")
	
	def __second_row_middle(self, col=st):
		"""
		"""
		selection = col.radio('PlotControll', ['Decktypen', 'Elo', 'Zeitstrahl', 'Tiers'],
						horizontal=True, label_visibility='collapsed')
		if selection == 'Elo':
			fig =  self.vis.box_violin_plot(self.df, 'Type', 'Elo', 'Eloverteilung pro Decktyp', 'Elo')
			col.pyplot(fig, transparent=True)
		elif selection == 'Decktypen':
			tmp = self.df[['Type', 'Deck']].groupby('Type').count().reset_index()
			fig = self.vis.ploty_bar(tmp, 'Type', 'Deck', False, title='Verteilung der Decks auf die Typen')
			col.plotly_chart(fig, tranparent=True)
		elif selection == 'Zeitstrahl':
			with col:
				vit.timeline_for_decks()
		elif selection == 'Tiers':
			fig = self.__plot_deck_desity(np.array(self.df['Elo'].values).squeeze())
			#st.markdown("<h4 style='text-align: center; color: withe;' >Elo-Verteilung</h4>", unsafe_allow_html=True)
			col.pyplot(fig, transparent=True)
		
		

	def __second_row_right(self, col:st):
		"""
		"""
		selection = col.radio('TypeControl', list(self.df['Type'].unique()), horizontal=True, 
			label_visibility='collapsed')
		
		cols = col.columns([.5, 1, .5])
		tmp = self.df[self.df['Type']==selection].copy().reset_index(drop=True)
		idx = tmp['Elo'].argmax()
		vit.load_and_display_image(pos=cols[1],
								path=f'./Deck_Icons/{tmp.at[idx, "Deck"]}.png',
								subtext=f"{tmp.at[idx, 'Deck']} {tmp.at[idx, 'Elo']}")
		n = len(tmp)
		cols[1].markdown(f"<h5 style='text-align: center; color: withe;' >Decks {n}</h5>", unsafe_allow_html=True)

							
	def __plot_deck_desity(self, values):  # sourcery skip: extract-duplicate-method  # sourcery skip: extract-duplicate-method
		'''
		
		'''
		Label = ['Kartenstapel','Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0']
		H, bins = self.__do_histogram(values, b=10, d=False)
		
		mean = np.mean(values)
		std = np.std(values)
		
		x = np.linspace(mean-4*std, mean+4*std, 500)
		a = 0.3

		fig = plt.figure(figsize=(6,6))
		fig.set_figheight(10)
		fig.set_figwidth(15)
		
		# create grid for different subplots
		spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0, height_ratios=[6, 1])
		axs = fig.add_subplot(spec[0])
		#axs.set_title('Elo-Verteilung', fontsize=20, color=self.axis_color)
		axs1 = fig.add_subplot(spec[1])
		
		axs2 = axs.twinx()
		# 68% Intervall
		axs2.fill_between(np.linspace(mean-std, mean+std, 100), self.__gauss(np.linspace(mean-std, mean+std, 100), mean, std), color='b', label='68% Interv.', alpha=a)
		# 95% Intervall
		axs2.fill_between(np.linspace(mean-2*std, mean-std, 100), self.__gauss(np.linspace(mean-2*std, mean-std, 100), mean, std), color='g', label='95% Interv.', alpha=a)
		axs2.fill_between(np.linspace(mean+std, mean+2*std, 100), self.__gauss(np.linspace(mean+std, mean+2*std, 100), mean, std), color='g', alpha=a)
		# Signifikanter Bereich
		axs2.fill_between(np.linspace(mean-4*std, mean-2*std, 100), self.__gauss(np.linspace(mean-4*std, mean-2*std, 100), mean, std), color='r', label='Sign. Bereich', alpha=a)
		axs2.fill_between(np.linspace(mean+2*std, mean+4*std, 100), self.__gauss(np.linspace(mean+2*std, mean+4*std, 100), mean, std), color='r', alpha=a)
		# Verteilungsfunktion
		axs2.plot(x, self.__gauss(x, mean, std), label='PDF', color='gray')
		#
		axs2.plot([mean+2*std, mean+2*std], [0, self.__gauss(mean+2*std, mean, std)], color='gray')
		axs2.text(mean+2*std, self.__gauss(mean+2*std, mean, std),'%.f' % (mean+2*std), fontsize=15, color=self.axis_color)

		axs2.plot([mean+std, mean+std], [0, self.__gauss(mean+std, mean, std)], color='gray')
		axs2.text(mean+std, self.__gauss(mean+std, mean, std),'%.f' % (mean+std), fontsize=15, color=self.axis_color)

		axs2.plot([mean, mean], [0, self.__gauss(mean, mean, std)], color='gray')
		axs2.text(mean, self.__gauss(mean, mean, std), '%.f' % (mean), fontsize=15, color=self.axis_color)
		
		axs2.plot([mean-std, mean-std], [0, self.__gauss(mean-std, mean, std)], color='gray')
		axs2.text((mean-std)-20, self.__gauss(mean-std, mean, std),'%.f' % (mean-std), fontsize=15, color=self.axis_color)

		axs2.plot([mean-2*std, mean-2*std], [0, self.__gauss(mean-2*std, mean, std)], color='gray')
		axs2.text((mean-2*std)-20, self.__gauss(mean-2*std, mean, std),'%.f' % (mean-2*std), fontsize=15, color=self.axis_color)

		# Data
		axs.bar(bins, H, abs(bins[1]-bins[0]), alpha=0.65, color='gray', label='Deck Histogram')

		# Layout
		axs.set_ylabel('Anzahl der Decks', fontsize=20, color=self.axis_color)
		axs2.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=20, color=self.axis_color)
		axs.grid()
		axs.legend(loc='upper left', fontsize=15)
		axs2.legend(loc='upper right', fontsize=15)
		axs2.set_ylim([0, 1.1*np.max(self.__gauss(x, mean, std))])
		axs.set_xlim([mean-4*std, mean+4*std])
		
		# markiere die bereiche
		axs1.fill_between([mean-4*std, mean-2*std], [1, 1], color='r', alpha=a)
		axs1.fill_between([mean-2*std, mean-std], [1, 1], color='g', alpha=a)
		axs1.fill_between([mean-std, mean+std], [1, 1], color='b', alpha=a)
		axs1.fill_between([mean+std, mean+2*std], [1, 1], color='g', alpha=a)
		axs1.fill_between([mean+2*std, mean+4*std], [1, 1], color='r', alpha=a)
		# abgrenzung der bereiche
		axs1.plot([mean-2*std, mean-2*std], [0, 1], color='gray')
		axs1.plot([mean-std, mean-std], [0, 1], color='gray')
		axs1.plot([mean, mean], [0, 1], color='gray')
		axs1.plot([mean+std, mean+std], [0, 1], color='gray')
		axs1.plot([mean+2*std, mean+2*std], [0, 1], color='gray')
		# text in den bereichen
		offset1 = 25
		offset = 20
		
		axs1.xaxis.label.set_color(self.axis_color)
		axs1.yaxis.label.set_color(self.axis_color)
		axs2.xaxis.label.set_color(self.axis_color)
		axs2.yaxis.label.set_color(self.axis_color)
		
		axs1.spines['bottom'].set_color(self.axis_color)
		axs1.spines['left'].set_color(self.axis_color)
		axs1.spines['top'].set_color(self.axis_color)
		axs1.spines['right'].set_color(self.axis_color)
		axs1.tick_params(colors=self.axis_color, which='both')
		
		axs2.spines['bottom'].set_color(self.axis_color)
		axs2.spines['left'].set_color(self.axis_color)
		axs2.spines['top'].set_color(self.axis_color)
		axs2.spines['right'].set_color(self.axis_color)
		axs2.tick_params(colors=self.axis_color, which='both')
		
		axs1.text(mean-3*std-offset, 0.65, Label[0], fontsize=20, color=self.axis_color)
		N = len(values[(values>=mean-4*std)&(values<mean-2*std)])
		axs1.text(mean-3*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color=self.axis_color)
		
		axs1.text(mean-1.5*std-offset, 0.65, Label[1], fontsize=20, color=self.axis_color)
		N = len(values[(values>=mean-2*std)&(values<mean-std)])
		axs1.text(mean-1.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color=self.axis_color)
		
		axs1.text(mean-0.5*std-offset, 0.65, Label[2], fontsize=20, color=self.axis_color)
		N = len(values[(values>=mean-std)&(values<mean)])
		axs1.text(mean-0.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color=self.axis_color)
		
		axs1.text(mean+0.5*std-offset, 0.65, Label[3], fontsize=20, color=self.axis_color)
		N = len(values[(values>=mean)&(values<mean+std)])
		axs1.text(mean+0.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color=self.axis_color)
		
		axs1.text(mean+1.5*std-offset, 0.65, Label[4], fontsize=20, color=self.axis_color)
		N = len(values[(values>=mean+std)&(values<mean+2*std)])
		axs1.text(mean+1.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color=self.axis_color)
		
		axs1.text(mean+3*std-offset, 0.65, Label[5], fontsize=20, color=self.axis_color)
		N = len(values[(values>=mean+2*std)&(values<mean+4*std)])
		axs1.text(mean+3*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color=self.axis_color)
		
		axs1.set_xlim([mean-4*std, mean+4*std])
		axs1.set_xlabel('ELO', fontsize=20, color=self.axis_color)
		axs1.set_ylim([0, 1])
		axs1.set(yticklabels=[])  # remove the tick labels
		axs1.tick_params(left=False)  # remove the ticks
		axs1.grid()
		axs1.tick_params(axis='x', labelsize=15, color=self.axis_color)
		axs1.tick_params(axis='y', labelsize=15, color=self.axis_color) 
		axs2.tick_params(axis='y', labelsize=15, color=self.axis_color) 
		axs.tick_params(axis='y', labelsize=15, color=self.axis_color) 
		return fig
	
	def __gauss(self, x, m, s):
		'''
		
		'''
		return 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2/(4*s**2))

	def __do_histogram(self, data, b=10, d=True):
		'''
		
		'''
		counts, bins = np.histogram(data, b, density=d)

		n = np.size(bins)
		cbins = np.zeros(n-1)

		for ii in range(n-1):
			cbins[ii] = (bins[ii+1]+bins[ii])/2

		return counts, cbins