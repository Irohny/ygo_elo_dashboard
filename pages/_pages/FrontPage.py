import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

class FrontPage:
	'''
	Class for generating the firste page of the dashboard with
	some major statistics
	'''
	def __init__(self, df, hist_cols):
		self.__build_page_layout(df, hist_cols)

	def __build_page_layout(self, df, hist_cols):
		n = len(df['Deck'].unique())
		# layout
		self.__first_kpi_row(df, n)
		st.markdown("----")
		self.__second_kpi_row(df, hist_cols)
		st.markdown("----")
		self.__third_kpi_row(df,n)
		st.markdown("----")
		self.__vergleiche_stile(df)

	def __first_kpi_row(self, df, n):
		# layout
		cols_layout = st.columns([2,2,2,2])
		with cols_layout[0]:
			st.header(f"Decks in Wertung:")
			st.header(n)
				
		with cols_layout[1]:
			st.header(f"Gesamtzahl Matches:")
			st.header(int(df['Matches'].sum()/2))
				
		with cols_layout[2]:
			st.header(f"Gesamtzahl Spiele:")
			st.header(int((df['Siege'].sum()+df['Remis'].sum()+df['Niederlage'].sum())/2))

		with cols_layout[3]:
			st.header(f"Anzahl Spieler:")
			st.header(len(df['Owner'].unique()))

	def __second_kpi_row(self, df, hist_cols):
		df_max = df[hist_cols+['Elo']].agg(['idxmax', 'max'])
		df_min = df[df[hist_cols+['Elo']]>0][hist_cols+['Elo']].agg(['idxmin', 'min'])
		cols_layout = st.columns([3,3,3,3])
		with cols_layout[0]:
			st.header('Höchste Elo')	
		with cols_layout[1]:
			st.header('Aktuell beste Elo')
		with cols_layout[2]:
			st.header('Niedgriste Elo')
		with cols_layout[3]:
			st.header('Aktuelle niedrigste Elo')
		
		metric_cols = st.columns([1,2,1,2,1,2,1,2])
		with metric_cols[0]:
			deck = df.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img)
		with metric_cols[1]:
			st.metric(f"", value=f"{df.at[int(df_max.at['idxmax', df_max.loc['max', :].idxmax()]), 'Deck']}", delta=f"{int(df_max.loc['max', :].max())}")

		with metric_cols[2]:
			deck = df.at[int(df_max.at['idxmax', 'Elo']), 'Deck']
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img)
		with metric_cols[3]:
			st.metric(f"", value=f"{df.at[int(df_max.at['idxmax', 'Elo']), 'Deck']}", delta=f"{int(df_max.at['max', 'Elo'])}")

		with metric_cols[4]:
			deck = df.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img)
		with metric_cols[5]:
			st.metric(f"", value=f"{df.at[df_min.at['idxmin', df_min.loc['min', :].idxmin()], 'Deck']}", delta=f"{int(df_min.loc['min', :].min())}")

		with metric_cols[6]:
			deck = df.at[df_min.loc['idxmin', 'Elo'], 'Deck']
			img = plt.imread(f'./Deck_Icons/{deck}.png')
			st.image(img)
		with metric_cols[7]:
			st.metric(f"", value=f"{df.at[df_min.loc['idxmin', 'Elo'], 'Deck']}", delta=f"{int(df_min.loc['min', 'Elo'].min())}")

	def __third_kpi_row(self, df, n):
		types = list(df['Type'].unique())
		st.header('Beste Decks der Kategorien:')
		colst = st.columns(len(types))
		for k in range(len(types)):
				with colst[k]:
					tmp = df[df['Type']==types[k]].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
					idx = tmp.at[0, 'Deck']
					d = tmp.at[0, 'Elo']
					st.header(types[k])
					st.metric('', value=idx, delta=f"{d}")
					img = plt.imread(f'./Deck_Icons/{idx}.png')
					st.image(img)
					

		colsk = st.columns(6)
		for i, feat in enumerate(types):
				with colsk[i]:
					n = len(df[df['Type']==feat])
					st.metric('Anzahl Decks ' + feat, value=n)

	def __vergleiche_stile(self, df):
		'''
		'''
		types = df['Type'].unique()
		histo, viobox, c3 = st.columns([1, 1, 1])
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
		
		with c3:
			fig = self.__plot_deck_desity(np.array(df['Elo'].values).squeeze())
			st.pyplot(fig, transparent=True)

	@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
	def __plot_deck_desity(self, values):
      
		Label = ['Kartenstapel','Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0']
		H, bins = self.__do_histogram(values, b=10, d=False)
		
		mean = np.mean(values)
		std = np.std(values)
		
		x = np.linspace(mean-4*std, mean+4*std, 500)
		a = 0.3

		fig = plt.figure(figsize=(8,8))
		fig.set_figheight(10)
		fig.set_figwidth(15)
		
		# create grid for different subplots
		spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0, height_ratios=[6, 1])
		axs = fig.add_subplot(spec[0])
		axs.set_title('Elo-Verteilung', fontsize=20, color='white')
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
		return 1/(s*np.sqrt(2*np.pi))*np.exp(-(x-m)**2/(4*s**2))

	def __do_histogram(self, data, b=10, d=True):
		counts, bins = np.histogram(data, b, density=d)

		n = np.size(bins)
		cbins = np.zeros(n-1)

		for ii in range(n-1):
			cbins[ii] = (bins[ii+1]+bins[ii])/2

		return counts, cbins