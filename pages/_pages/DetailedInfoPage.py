import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pages._pages.Visualization import Visualization

class DetailedInfoPage:
    '''
    Class for a page with a filter to choose a deck an see all
    statistics of this deck as well as its historics results
    '''

    def __init__(self, df, hist_cols):
        '''
                Init-Method
                :param df: dataframe with elo data
                :param hist_cols: column names with old elo data
        '''
        # set dashboard colors
        self.back_color = "#00172B"
        self.filed_color = "#0083B8"
        # build layout
        self.vis = Visualization()
        self.__build_page_layout(df, hist_cols)

    def __build_page_layout(self, df, hist_cols):
        '''
                Method for creating the layout of the detailed info sub page.
                :param df: dataframe with elo data
                :param hist_cols: list with column names of old elo data
        '''
        # set data format and display a select box for choosing the deck of interest
        deck_ops = df['Deck'].astype(str).to_numpy()
        deck_ops = np.append(np.array(['']), deck_ops)
        deck = st.selectbox('Wähle ein deck', options=deck_ops, index=0)
        # if deck was choosen display the stats
        if not deck:
            st.stop()

        # find index of the choosen deck and get elo, rank and percentaage of weaker decks
        idx_deck = int(df[df['Deck'] == deck].index.to_numpy())
        tmp_df = df.copy()
        tmp_df = tmp_df[['Deck', 'Elo']].sort_values(
            by=['Elo'], ascending=False).reset_index(drop=True)
        platz = int(tmp_df[tmp_df['Deck'] == deck].index.to_numpy())+1
        percentage = (int((len(df)-platz+1)/len(df)*1000)/10)
        # Layout of deck KPI's, organisation in two rows
        # first row with deck name, tier, Icon, type, win/lose plot and stats spider
        header_col = st.columns([1, 1.5, 3, 2])
        # display deck name, icon and type
        header_col[0].header(
            f'{df.at[idx_deck, "Tier"]},    {df.at[idx_deck, "Type"]}')
        try:
            img = plt.imread(f'./Deck_Icons/{deck}.png')
            header_col[0].image(img, use_column_width=True)
        except Exception:
            header_col[0].error('Kein Bild vorhanden')

        # win rate plot 
        values = df.loc[idx_deck, ['Siege', 'Remis', 'Niederlage']].to_list()
        fig = self.vis.plotly_gauge_plot(100*values[0]/sum(values)//1)
        header_col[1].plotly_chart(fig, use_container_width=True)

        # stats part
        dy = df.loc[idx_deck, hist_cols+['Elo']]
        dy = dy[dy>0].reset_index(drop=True)
        fig = self.vis.plot_metric(label="Aktuelle Elo", 
                                value=df.at[idx_deck, 'Elo'],
                                x_data=hist_cols+['Elo'],
                                y_data=(dy-.95*dy.min()).values,
                                show_graph=True,
                                color_graph='rgba(0, 104, 201, 0.2)')
        header_col[2].plotly_chart(fig, 
                                use_container_width=True,
                                theme="streamlit")
        inside = header_col[2].columns([1, 1, 1])
        inside[1].metric('Matches', int(
            df.at[idx_deck, 'Matches']), f"{int(df.at[idx_deck, 'Siege']+df.at[idx_deck, 'Remis']+df.at[idx_deck, 'Niederlage'])} Spiele")
        inside[0].metric("Aktueller Rang", value=f'{platz}', delta=f'Besser als {percentage}%')

        ds = pd.Series(df.loc[idx_deck, hist_cols+['Elo']].values)
        ds = ds[ds > 0].reset_index(drop=True)
        inside[2].metric("Beste Elo", 
                        value=f"{int(df.loc[idx_deck, hist_cols+['Elo']].max())}",
                        delta=f"Schlechteste Elo: {int(ds.min())}", delta_color='inverse')
        # disply stats spider plot
        categories = ['Attack', 'Control', 'Recovery',
                        'Consistensy', 'Combo', 'Resilience']
        fig = self.vis.make_spider_plot(df.loc[idx_deck, categories].astype(int).to_list())
        header_col[3].plotly_chart(fig, theme="streamlit", use_container_width=True)

        # second row for displaying elo KPI's like rank, titles and historic elo ratings
        col = st.columns([2,5])
        # wanderpokal
        wanderpokal = ''.join(':trophy: ' for _ in range(int(df.at[idx_deck, 'Meisterschaft']['Win']['Wanderpokal'])))
        col[0].markdown(f'Wanderpokal {wanderpokal}')
        # local wins
        localwin = ''.join(':medal: ' for _ in range(int(df.at[idx_deck, 'Local Win'])))
        col[0].markdown(f'Local Win {localwin}')
        #
        funpokal = ''.join(':star: ' for _ in range(int(df.at[idx_deck, 'Meisterschaft']['Win']['Fun Pokal'])))
        col[0].markdown(f'Fun Pokal {funpokal}')
        #
        localtop= ''.join(':star: ' for _ in range(int(df.at[idx_deck,'Local Top'])))
        col[0].markdown(f'Local Top {localtop}')
        fig = self.__line_plot(df, hist_cols, idx_deck)
        col[1].plotly_chart(fig, theme=None, use_container_width=True)
        
            

    def __line_plot(self, df, hist_cols, idx):
        '''
        Method for creating a lineplot of the elo points of the last year
        :param df: dataframe with historic elo
        :param hist_cols: list wit historic columns
        :param idx: index of the current deck
        :return fig: plotly figure object
        '''
        # generate plot dataframe with elo, date and mean of the last year
        df_plot = pd.DataFrame(columns=['Elo'])
        df_plot['Elo'] = df.loc[idx, hist_cols[-4:]+['Elo']].values
        df_plot['Datum'] = hist_cols[-4:]+['Elo']
        df_plot = df_plot[df_plot['Elo'] > 0].reset_index(drop=True)
        df_plot['Mittelw.'] = df_plot['Elo'].mean()
        # generate plotly figure and plot as well as layout of the figure
        return self.vis.lineplot(df_plot["Datum"], df_plot["Elo"], df_plot["Mittelw."])


