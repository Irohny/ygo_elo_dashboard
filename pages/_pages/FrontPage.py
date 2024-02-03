import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import gridspec
from streamlit_timeline import timeline
import datetime
from pages._pages.Visualization import Visualization

class FrontPage:
	'''
	Class for generating the firste page of the dashboard with
	some major statistics
	'''
	def __init__(self, df, tdf, hist_cols):
		'''
		Init-Method
		:param df: dataframe with elo data
		:param tdf: dataframe with tournament data
		:param hist_cols: list of column names with old elo
		'''
		# build page layout
		self.axis_color = 'black'
		self.vis = Visualization()
		self.__build_page_layout(df, tdf, hist_cols)

	def __build_page_layout(self, df, tdf, hist_cols):
		'''
		Method for layouting the the front page
		:param df: dataframe with elo data
		:param tdf: dataframe with tournament data
		:param hist_cols: list with column names of old elo data 
		'''
		# get number of all decks
		n = len(df['Deck'].unique())
		st.title(':trophy: YuGiOh! Elo-Dashboard :trophy:')
		overview, classes, timeline = st.tabs(['Überblick', 'Deckklassen', 'Deckvergleiche'])
		# first tab on front page
		with overview:
			self.__first_kpi_row(df, tdf, n)
			self.__second_kpi_row(df, tdf, hist_cols)
		# second tab on front page
		with classes:
			self.__third_kpi_row(df,n)
			# fourth row: histograms and box-violin plots of elo distributions
			self.__vergleiche_stile(df)
		# third tab on front page
		with timeline:
			cols = st.columns([.2, 1,1, .2])
			with cols[1]:
				st.title('Top 5 Zeitstrahl')
				self.__time_line_widget(df, hist_cols)
			with cols[2]:
				st.markdown('')
				st.markdown('')
				fig = self.__plot_deck_desity(np.array(df['Elo'].values).squeeze())
				st.markdown("<h4 style='text-align: center; color: withe;' >Elo-Verteilung</h4>", unsafe_allow_html=True)
				st.pyplot(fig, transparent=True)
			
	def __first_kpi_row(self, df, tdf, n):
		'''
		Method for displaying the first row of the front page KPI's
		Show number of player, games, deck
		:param df: dataframe with elo/deck data
		:param tdf: dataframe with tournament data
		:param n: number of all decks 
		'''
		cols = st.columns([2,2,2,2,2,2])
		# first column with number of decks
		cols[0].metric(":flower_playing_cards: Decks in Wertung:", n)
		# second columns with number of matches
		cols[1].metric(":crossed_swords: Gesamtzahl Matches:", int(df['Matches'].sum()/2))
		# third column with number of games
		cols[2].metric(":crossed_swords: Gesamtzahl Spiele:", int((df['Siege'].sum()+df['Remis'].sum()+df['Niederlage'].sum())/2))
		# fourth column with number of player
		cols[3].metric(":bust_in_silhouette: Anzahl Spieler:", len(df['Owner'].unique()))
		# fourth column with number of player
		cols[4].metric(':bank: Turnierteilnahmen', len(tdf))
		# fourth column with number of player
		cols[5].metric(':medal: Top-Rate', f'{np.round(len(tdf[tdf["Standing"]=="Top"])/len(tdf)*100,2)}%')

	def __second_kpi_row(self, df, tdf, hist_cols):
		'''
		Method for displaying the second KPI row of the dashboard frontpage
		:param df: dataframe with elo/deck data
		:param tdf: dataframe with tournament data
		:param hist_cols: list of columns names with old elo
		'''
		# get best and worst decks in the data
		df_max = df[hist_cols+['Elo']].agg(['idxmax', 'max'])
		df_min = df[df[hist_cols+['Elo']]>0][hist_cols+['Elo']].agg(['idxmin', 'min'])
		# define layout columns with header for the following icons and metrics
		cols = st.columns([2,2,4,2,2])

		# first column: all over best elo
		cols[0].header('Höchste Elo:')
		deck = df.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']
		elo = int(df_max.loc['max', :].max())
		self.__deck_image_plot(cols[0], deck, elo)
		
		# second column: actual best elo	
		
		cols[1].header('Beste Elo:')
		deck = df.at[int(df_max.at['idxmax', 'Elo']), 'Deck']
		elo = int(df_max.at['max', 'Elo'])
		self.__deck_image_plot(cols[1], deck, elo)

		# tournament standings
		tmp = tdf[['Deck', 'Win', 'Loss', 'Draw', 'Standing']]
		tops = tdf[tdf['Standing']=='Top'].groupby(by='Deck').count()
		counts = tdf[['Standing', 'Deck']].groupby(by='Deck').count()
		tmp['Standing'] = tmp['Standing'].apply(self.__tournament_points)
		tmp['Points'] = 3*tmp['Win'] + tmp['Draw'] + 5*tmp['Standing']
		tmp = tmp.groupby(by='Deck').sum()
		tmp.sort_values(by='Points', ascending=False, inplace=True)
		#tmp.set_index('Deck', inplace=True)
		cols[2].header("Turnierdecks:")
		for pos, idx in enumerate(tmp.index):
			if pos>9:
				continue
			if idx in tops.index.to_list():
				n_tournaments = counts.at[idx, 'Standing']
				top_rate = np.round(100*tops.at[idx, 'Standing']/n_tournaments,2)
			elif idx in counts.index.to_list():
				n_tournaments = counts.at[idx, 'Standing']
				top_rate = 0
			else:
				n_tournaments = 0
				top_rate = 0
			tmp.at[idx, 'Top Rate'] = f"{top_rate}%"
			tmp.at[idx, 'Platz'] = pos+1
			tmp.at[idx, 'Turniere'] = n_tournaments
		tmp = tmp[tmp['Platz']<=10]
		tmp.reset_index(inplace=True)
		tmp.rename(columns={'Win':'S', 'Draw':'U', 'Loss':'N', 'Points':'Punkte'}, inplace=True)
		cols[2].dataframe(tmp[['Platz', 'Deck','Punkte','S', 'U', 'N', 'Turniere', 'Top Rate']],
			hide_index=True)
			
		# third column: all over worst elo
		cols[3].header('geringste Elo:')
		deck = df.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']
		img = plt.imread(f'./Deck_Icons/{deck}.png')
		elo = int(df_min.loc['min', :].min()) 
		self.__deck_image_plot(cols[3], deck, elo)

		# fourth column: actual worst elo
		cols[4].header('niedrigste Elo:')
		deck = df.at[df_min.loc['idxmin', 'Elo'], 'Deck']
		elo = int(df_min.loc['min', 'Elo'].min())
		self.__deck_image_plot(cols[4], deck, elo)
		
	def __deck_image_plot(self, col, deck, elo):
		'''
		:param col: streamlit column element
		'''
		try:
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			col.image(img, use_column_width=True)
		except Exception:
			col.error('No image')
		col.subheader(f"{deck} {elo}")

	def __tournament_points(self, x):
		"""
		"""
		coding = {'Top':3, 'Teilnahme':1}
		return coding[x]
	
	def __time_line_widget(self, df, hist_cols):
		'''
		'''
		time_json = {'title':{"text": {
          					"headline": "Top 5",
          					"text": "Die besten Decks im Vergleich"
        					}},
					'events':[]}
		
		for date in hist_cols:
			# setup date
			datum = {"year":str(date[3:]),
					"month":str(int(date[:2]))}
			# set up placements 
			test = df[[date, 'Deck']].copy().sort_values(by=date, ascending=False).reset_index(drop=True)
			text = {'text': f"<p>{test.at[0, 'Deck']} {int(test.at[0, date])}<p>\
							<p>{test.at[1, 'Deck']} {int(test.at[1, date])}<p>\
							<p>{test.at[2, 'Deck']} {int(test.at[2, date])}<p>\
							<p>{test.at[3, 'Deck']} {int(test.at[3, date])}<p>\
							<p>{test.at[4, 'Deck']} {int(test.at[4, date])}<p>",
					'headline':""}
			event = {"start_date": datum,
        			"text":text}
			time_json['events'].append(event)
		timeline(time_json, height=410)
		
		


	def __third_kpi_row(self, df, n):
		'''
		Method for displaying the third KPI row of the front page of the elo dashboard
		:param df: dataframe with elo/deck data
		:param n: number of all decks
		'''
		# get list of deck types
		types = list(df['Type'].unique())
		# display row header and define layout for best categorie dekcs 
		#st.header('Beste Decks der Kategorien:')
		colst = st.columns(len(types))
		# loop over all rows and categories
		for k in range(len(types)):
				# plot in this colum the name, icon, elo of the best deck in the categorie
				with colst[k]:
					tmp = df[df['Type']==types[k]].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
					idx = tmp.at[0, 'Deck']
					d = tmp.at[0, 'Elo']
					n = len(tmp)
					st.header(f"{types[k]}")
					img = plt.imread(f'./Deck_Icons/{idx}.png')
					st.image(img, use_column_width=True)
					st.markdown(f"<h5 style='text-align: center; color: withe;' >{idx} {d}</h5>", unsafe_allow_html=True)
					st.markdown(f"<h5 style='text-align: center; color: withe;' >Decks {n}</h5>", unsafe_allow_html=True)

	def __vergleiche_stile(self, df):  # sourcery skip: extract-duplicate-method
		'''
		'''
		types = df['Type'].unique()
		histo, viobox = st.columns([1, 1])

		
		# histogram plot part
		n_types = len(types)
		fig, axs = plt.subplots(1, figsize=(8,5))
		n_decks = np.zeros(n_types)
		for k in range(n_types):
			n_decks[k] = len(df[df['Type']==types[k]])
		axs.bar(range(n_types), n_decks, color='gray')
		axs.set_xticks(range(n_types))
		axs.set_xticklabels(types, rotation=-45, color=self.axis_color)
		axs.set_ylabel('Anzahl Decks', color=self.axis_color)
		axs.grid()
		axs.set_title('Verteilung der Decks auf die Typen', color=self.axis_color)
		axs.spines['bottom'].set_color(self.axis_color)
		axs.spines['left'].set_color(self.axis_color)
		axs.tick_params(colors=self.axis_color, which='both')
		histo.pyplot(fig, transparent=True)
				
		# violin plot part
		fig, axs = plt.subplots(1, figsize=(8,5))
		for idx, tmp_type in enumerate(types):
			axs.violinplot(df[df['Type']==tmp_type]['Elo'], positions=[idx], showmeans=True, showextrema=False, showmedians=False)
			axs.boxplot(df[df['Type']==tmp_type]['Elo'], positions=[idx])
		axs.set_title('Eloverteilung pro Decktyp', color=self.axis_color)
		axs.set_ylabel('Elo-Rating', color=self.axis_color)
		axs.set_xticks(range(n_types))
		axs.set_xticklabels(types, rotation=-45, color=self.axis_color)
		axs.grid()
		
		axs.spines['bottom'].set_color(self.axis_color)
		axs.spines['left'].set_color(self.axis_color)
		axs.tick_params(colors=self.axis_color, which='both')
		viobox.pyplot(fig, transparent=True)
			
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