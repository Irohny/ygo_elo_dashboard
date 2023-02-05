import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class DetailedInfoPage:
	'''
	Class for a page with a filter to choose a deck an see all
	statistics of this deck as well as its historics results
	'''
	def __init__(self, df, hist_cols):
		self.back_color = "#00172B"
		self.filed_color = "#0083B8"
		self.__build_page_layout(df, hist_cols)

	def __build_page_layout(self, df, hist_cols):
		'''
		'''
		deck_ops = df['Deck'].astype(str).to_numpy()
		deck_ops = np.append(np.array(['']), deck_ops)
		deck = st.selectbox('Wähle ein deck', options=deck_ops, index=0)
		if len(deck) > 0:
			# calculations
			idx_deck = int(df[df['Deck']==deck].index.to_numpy())
			tmp_df = df.copy()
			tmp_df = tmp_df[['Deck', 'Elo']].sort_values(by=['Elo'], ascending=False).reset_index(drop=True)
			platz = int(tmp_df[tmp_df['Deck']==deck].index.to_numpy())+1
			percentage = (int((len(df)-platz+1)/len(df)*1000)/10)
			# Layout
			header_col = st.columns([1.2, 4, 2, 2])
			with header_col[0]:
				st.header(f'{df.at[idx_deck, "Tier"]},    {df.at[idx_deck, "Type"]}')
				img = plt.imread(f'./Deck_Icons/{deck}.png')
				st.image(img, use_column_width=True)
			with header_col[2]:
				st.header(f"Matches: {int(df.at[idx_deck, 'Matches'])}, Spiele: {int(df.at[idx_deck, 'Siege']+df.at[idx_deck, 'Remis']+df.at[idx_deck, 'Niederlage'])}")
				fig = self.__semi_circle_plot(df.loc[idx_deck, ['Siege', 'Remis', 'Niederlage']].to_list())
				st.pyplot(fig, transparent=True)
			with header_col[3]:
				categories = ['Attack', 'Control', 'Recovery', 'Consistensy', 'Combo', 'Resilience']
				fig = self.__make_spider_plot(df.loc[idx_deck, categories].astype(int).to_list())
				st.plotly_chart(fig, theme="streamlit", use_container_width=True)
				
				
			deck_cols = st.columns([1,1,1,4])
			with deck_cols[0]:
					st.metric(f"Aktueller Rang", value=platz)
					st.metric(f"Besser als", value=f'{percentage}%')
					st.metric('Aktuelle Elo', df.at[idx_deck, 'Elo'], int(df.at[idx_deck, 'Letzte 3 Monate']))
					st.caption('Wanderpokal:   ' + ':star:'*int(df.at[idx_deck, 'Wanderpokal']))
					st.caption('Fun Pokal:     ' + ':star:'*int(df.at[idx_deck, 'Fun Pokal']))
					st.caption('Meisterschaft: ' + ':star:'*int(df.at[idx_deck, 'Meisterschaft']))
					st.caption('Liga Pokal:    ' + ':star:'*int(df.at[idx_deck, 'Liga Pokal']))
					
			with deck_cols[1]:
					st.metric(f"Beste Elo", value=f"{int(df.loc[idx_deck, hist_cols+['Elo']].max())}")
					ds = pd.Series(df.loc[idx_deck, hist_cols+['Elo']].values)
					ds = ds[ds>0].reset_index(drop=True)
					st.metric(f"Schlechteste Elo", value=f"{int(ds.min())}")
					st.metric('Gegner Stärke', int(df.at[idx_deck, 'dgp']))

			with deck_cols[2]:
					st.metric('Jahresänderung', value=int(df.loc[idx_deck, 'Letzte 12 Monate']))
					st.metric('Jahresmittel', value=int(df.at[idx_deck, 'Mean 1 Year']), delta=int(df.at[idx_deck, 'Std. 1 Year']))
					
			with deck_cols[3]:
					fig = self.__line_plot(df, hist_cols, idx_deck)
					st.plotly_chart(fig, theme=None, use_container_width=True)

			
	def __line_plot(self, df, hist_cols, idx):
		'''
		Method for creating a lineplot of the elo points of the last year
		:param df: dataframe with historic elo
		:param hist_cols: list wit historic columns
		:param idx: index of the current deck
		:return fig: plotly figure object
		'''
		df_plot = pd.DataFrame(columns=['Elo'])
		df_plot['Elo'] = df.loc[idx, hist_cols[-4:]+['Elo']].values
		df_plot['Datum'] = hist_cols[-4:]+['Elo']
		df_plot = df_plot[df_plot['Elo']>0].reset_index(drop=True)
		df_plot['Mittelw.'] = df_plot['Elo'].mean()
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=df_plot["Datum"], y=df_plot["Elo"], mode='lines+markers', name='Verlauf', line=dict(color="aqua", width=5)))
		fig.add_trace(go.Scatter(x=df_plot["Datum"], y=df_plot["Mittelw."], mode='lines+markers', name='Mittelw.', line=dict(color=self.filed_color, width=3)))
		fig.update_traces(textposition="bottom right")#
		fig.update_layout(font_size=15, template='plotly_white', title='Eloverlauf des letzen Jahres', xaxis_title='Datum', yaxis_title='Elo',
							paper_bgcolor=self.back_color, plot_bgcolor=self.back_color)
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
  						showlegend=False, title='Deck Eigenschaften',
						hovermode="x")
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
		fig = plt.figure(figsize=(7,4))
		ax = fig.add_subplot(1,1,1)
		#p = patches.Rectangle((left, bottom), width, height,fill=False, transform=ax.transAxes, clip_on=False)

		ax.pie(val, colors=colors, pctdistance=0.85, explode=explode)
		ax.text(-1.05, 1.1, f"N {int(val[2]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
		ax.text(-0.1, 1.1, f"U {int(val[1]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
		ax.text(0.65, 1.1, f"S {int(val[0]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
		ax.add_artist(plt.Circle((0, 0), 0.6, color=self.back_color))
		
		return fig
