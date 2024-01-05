import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import streamlit.components.v1 as components
from streamlit_timeline import timeline
import bar_chart_race as bcr
import base64
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
				#st.markdown('start')
				#rbc = self.__creat_racing_bar_chart(df, hist_cols).data
				#st.markdown('end')
				#start = rbc.find('base64,')+len('base64,')
				#end = rbc.find('">')
				#video = base64.b64decode(rbc[start:end])
				#st.video(video)
			with cols[2]:
				st.markdown('')
				st.markdown('')
				fig = self.__plot_deck_desity(np.array(df['Elo'].values).squeeze())
				st.markdown(f"<h4 style='text-align: center; color: withe;' >Elo-Verteilung</h4>", unsafe_allow_html=True)
				st.pyplot(fig, transparent=True)
			
	def __first_kpi_row(self, df, tdf, n):
		'''
		Method for displaying the first row of the front page KPI's
		Show number of player, games, deck
		:param df: dataframe with elo/deck data
		:param tdf: dataframe with tournament data
		:param n: number of all decks 
		'''
		cols_layout = st.columns([2,2,2,2,2,2])
		# first column with number of decks
		with cols_layout[0]:
			st.metric(f":flower_playing_cards: Decks in Wertung:", n)
		# second columns with number of matches
		with cols_layout[1]:
			st.metric(f":crossed_swords: Gesamtzahl Matches:", int(df['Matches'].sum()/2))
		# third column with number of games
		with cols_layout[2]:
			st.metric(f":crossed_swords: Gesamtzahl Spiele:", int((df['Siege'].sum()+df['Remis'].sum()+df['Niederlage'].sum())/2))
		# fourth column with number of player
		with cols_layout[3]:
			st.metric(f":bust_in_silhouette: Anzahl Spieler:", len(df['Owner'].unique()))
		# fourth column with number of player
		with cols_layout[4]:
			st.metric(':bank: Turnierteilnahmen', len(tdf))
		# fourth column with number of player
		with cols_layout[5]:
			st.metric(':medal: Top-Rate', f'{np.round(len(tdf[tdf["Standing"]=="Top"])/len(tdf)*100,2)}%')

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
		cols_layout = st.columns([2,2,4,2,2])
		# first column: all over best elo
		with cols_layout[0]:
			st.header('Höchste Elo:')
			deck = df.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']
			elo = int(df_max.loc['max', :].max())
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img, use_column_width=True)
			st.subheader(f"{deck} {elo}")
		# second column: actual best elo	
		with cols_layout[1]:
			st.header('Beste Elo:')
			deck = df.at[int(df_max.at['idxmax', 'Elo']), 'Deck']
			elo = int(df_max.at['max', 'Elo'])
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img, use_column_width=True)
			st.subheader(f"{deck} {elo}")
		# tournament standings
		with cols_layout[2]:
			tmp = tdf[['Deck', 'Win', 'Loss', 'Draw', 'Standing']]
			tops = tdf[tdf['Standing']=='Top'].groupby(by='Deck').count()
			counts = tdf[['Standing', 'Deck']].groupby(by='Deck').count()
			tmp['Standing'] = tmp['Standing'].apply(self.__tournament_points)
			tmp['Points'] = 3*tmp['Win'] + tmp['Draw'] + 5*tmp['Standing']
			tmp = tmp.groupby(by='Deck').sum()
			tmp.sort_values(by='Points', ascending=False, inplace=True)
			#tmp.set_index('Deck', inplace=True)
			st.header("Turnierdecks:")
			inside = cols_layout[2].columns([1, 4, 1,1,1,1,1,2])
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
			st.dataframe(tmp[['Platz', 'Deck','Punkte','S', 'U', 'N', 'Turniere', 'Top Rate']],
				hide_index=True)
			
			
			#st.dataframe(df_plot.head(10), use_container_width=True)

		# third column: all over worst elo
		with cols_layout[3]:
			st.header('geringste Elo:')
			deck = df.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			elo = int(df_min.loc['min', :].min()) 
			st.image(img, use_column_width=True)
			st.subheader(f"{deck} {elo}")
		# fourth column: actual worst elo
		with cols_layout[4]:
			st.header('niedrigste Elo:')
			deck = df.at[df_min.loc['idxmin', 'Elo'], 'Deck']
			elo = int(df_min.loc['min', 'Elo'].min())
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img, use_column_width=True)
			st.subheader(f"{deck} {elo}")

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
		for i, date in enumerate(hist_cols):
			datum = {"year":"","month":""}
			text = {"headline": "","text": ""}
			
			datum['year'] = str(date[3:])
			datum['month'] = str(int(date[:2]))
			test = df[[date, 'Deck']].copy().sort_values(by=date, ascending=False).reset_index(drop=True)
			text['text'] = f"<p>{test.at[0, 'Deck']} {int(test.at[0, date])}<p>\
							<p>{test.at[1, 'Deck']} {int(test.at[1, date])}<p>\
							<p>{test.at[2, 'Deck']} {int(test.at[2, date])}<p>\
							<p>{test.at[3, 'Deck']} {int(test.at[3, date])}<p>\
							<p>{test.at[4, 'Deck']} {int(test.at[4, date])}<p>"
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

	def __vergleiche_stile(self, df):
		'''
		'''
		types = df['Type'].unique()
		histo, viobox = st.columns([1, 1])

		
		with histo:
			n_types = len(types)
			fig, axs = plt.subplots(1, figsize=(8,5))
			n_decks = np.zeros(n_types)
			for k in range(n_types):
				n_decks[k] = len(df[df['Type']==types[k]])
			axs.bar(range(n_types), n_decks, color='gray')
			axs.set_xticks(range(n_types))
			axs.set_xticklabels(types, rotation=-45, color='white')
			axs.set_ylabel('Anzahl Decks', color='white')
			axs.grid()
			axs.set_title('Verteilung der Decks auf die Typen', color='white')
			axs.spines['bottom'].set_color('white')
			axs.spines['left'].set_color('white')
			axs.tick_params(colors='white', which='both')
			st.pyplot(fig, transparent=True)
				
		with viobox:
			fig, axs = plt.subplots(1, figsize=(8,5))
			for idx, tmp_type in enumerate(types):
				axs.violinplot(df[df['Type']==tmp_type]['Elo'], positions=[idx], showmeans=True, showextrema=False, showmedians=False)
				axs.boxplot(df[df['Type']==tmp_type]['Elo'], positions=[idx])
			axs.set_title('Eloverteilung pro Decktyp', color='white')
			axs.set_ylabel('Elo-Rating', color='white')
			axs.set_xticks(range(n_types))
			axs.set_xticklabels(types, rotation=-45, color='white')
			axs.grid()
			
			axs.spines['bottom'].set_color('white')
			axs.spines['left'].set_color('white')
			axs.tick_params(colors='white', which='both')
			st.pyplot(fig, transparent=True)
			
	def __plot_deck_desity(self, values):
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
		#axs.set_title('Elo-Verteilung', fontsize=20, color='white')
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
		axs2.text(mean+2*std, self.__gauss(mean+2*std, mean, std),'%.f' % (mean+2*std), fontsize=15, color='white')

		axs2.plot([mean+std, mean+std], [0, self.__gauss(mean+std, mean, std)], color='gray')
		axs2.text(mean+std, self.__gauss(mean+std, mean, std),'%.f' % (mean+std), fontsize=15, color='white')

		axs2.plot([mean, mean], [0, self.__gauss(mean, mean, std)], color='gray')
		axs2.text(mean, self.__gauss(mean, mean, std), '%.f' % (mean), fontsize=15, color='white')
		
		axs2.plot([mean-std, mean-std], [0, self.__gauss(mean-std, mean, std)], color='gray')
		axs2.text((mean-std)-20, self.__gauss(mean-std, mean, std),'%.f' % (mean-std), fontsize=15, color='white')

		axs2.plot([mean-2*std, mean-2*std], [0, self.__gauss(mean-2*std, mean, std)], color='gray')
		axs2.text((mean-2*std)-20, self.__gauss(mean-2*std, mean, std),'%.f' % (mean-2*std), fontsize=15, color='white')

		# Data
		axs.bar(bins, H, abs(bins[1]-bins[0]), alpha=0.65, color='gray', label='Deck Histogram')

		# Layout
		axs.set_ylabel('Anzahl der Decks', fontsize=20, color='white')
		axs2.set_ylabel('Wahrscheinlichkeitsdichte', fontsize=20, color='white')
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
		
		axs1.xaxis.label.set_color('white')
		axs1.yaxis.label.set_color('white')
		axs2.xaxis.label.set_color('white')
		axs2.yaxis.label.set_color('white')
		
		axs1.spines['bottom'].set_color('white')
		axs1.spines['left'].set_color('white')
		axs1.spines['top'].set_color('white')
		axs1.spines['right'].set_color('white')
		axs1.tick_params(colors='white', which='both')
		
		axs2.spines['bottom'].set_color('white')
		axs2.spines['left'].set_color('white')
		axs2.spines['top'].set_color('white')
		axs2.spines['right'].set_color('white')
		axs2.tick_params(colors='white', which='both')
		
		axs1.text(mean-3*std-offset, 0.65, Label[0], fontsize=20, color='white')
		N = len(values[(values>=mean-4*std)&(values<mean-2*std)])
		axs1.text(mean-3*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
		
		axs1.text(mean-1.5*std-offset, 0.65, Label[1], fontsize=20, color='white')
		N = len(values[(values>=mean-2*std)&(values<mean-std)])
		axs1.text(mean-1.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
		
		axs1.text(mean-0.5*std-offset, 0.65, Label[2], fontsize=20, color='white')
		N = len(values[(values>=mean-std)&(values<mean)])
		axs1.text(mean-0.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
		
		axs1.text(mean+0.5*std-offset, 0.65, Label[3], fontsize=20, color='white')
		N = len(values[(values>=mean)&(values<mean+std)])
		axs1.text(mean+0.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
		
		axs1.text(mean+1.5*std-offset, 0.65, Label[4], fontsize=20, color='white')
		N = len(values[(values>=mean+std)&(values<mean+2*std)])
		axs1.text(mean+1.5*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
		
		axs1.text(mean+3*std-offset, 0.65, Label[5], fontsize=20, color='white')
		N = len(values[(values>=mean+2*std)&(values<mean+4*std)])
		axs1.text(mean+3*std-offset1, 0.25, '%.f Decks' % (N), fontsize=20, color='white')
		
		axs1.set_xlim([mean-4*std, mean+4*std])
		axs1.set_xlabel('ELO', fontsize=20, color='white')
		axs1.set_ylim([0, 1])
		axs1.set(yticklabels=[])  # remove the tick labels
		axs1.tick_params(left=False)  # remove the ticks
		axs1.grid()
		axs1.tick_params(axis='x', labelsize=15, color='white')
		axs1.tick_params(axis='y', labelsize=15, color='white') 
		axs2.tick_params(axis='y', labelsize=15, color='white') 
		axs.tick_params(axis='y', labelsize=15, color='white') 
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
	
	def __creat_racing_bar_chart(self, df, hist_cols):
		rancing_df = []
		for col in hist_cols:
			tmp = df[['Deck', col]].copy()
			tmp.rename(columns={col:'Elo'}, inplace=True)
			tmp['Date'] = pd.to_datetime(f'{col[2:]}-{col[:2]}-01')
			rancing_df.append(tmp)
		tmp = df[['Deck', 'Elo']].copy()
		tmp['Date'] = pd.to_datetime(datetime.datetime.now())
		rancing_df.append(tmp)
		rancing_df = pd.concat(rancing_df).reset_index(drop=True)
		
		rancing_df = rancing_df.pivot(index = "Date", columns = "Deck", values = "Elo").reset_index().rename_axis(None, axis=1)
		rancing_df.fillna(0, inplace=True)
		rancing_df.set_index("Date", inplace = True)
		return bcr.bar_chart_race(df = rancing_df, 
			    				title = "Elo", 
								n_bars=10,
								bar_size=.9,
								dpi=75,
    							cmap='dark12',
								figsize=(16, 10),
								period_length=1500,
								steps_per_period=5,
								bar_label_size=15,
    							tick_label_size=15,
								scale='log',)