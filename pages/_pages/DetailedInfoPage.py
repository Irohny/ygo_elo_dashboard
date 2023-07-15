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
        '''
                Init-Method
                :param df: dataframe with elo data
                :param hist_cols: column names with old elo data
        '''
        # set dashboard colors
        self.back_color = "#00172B"
        self.filed_color = "#0083B8"
        # build layout
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
        if len(deck) > 0:
            # find index of the choosen deck and get elo, rank and percentaage of weaker decks
            idx_deck = int(df[df['Deck'] == deck].index.to_numpy())
            tmp_df = df.copy()
            tmp_df = tmp_df[['Deck', 'Elo']].sort_values(
                by=['Elo'], ascending=False).reset_index(drop=True)
            platz = int(tmp_df[tmp_df['Deck'] == deck].index.to_numpy())+1
            percentage = (int((len(df)-platz+1)/len(df)*1000)/10)
            # Layout of deck KPI's, organisation in two rows
            # first row with deck name, tier, Icon, type, win/lose plot and stats spider
            header_col = st.columns([1.6, 1.6, 2, 2, 2.1])
            # display deck name, icon and type
            with header_col[0]:
                st.header(
                    f'{df.at[idx_deck, "Tier"]},    {df.at[idx_deck, "Type"]}')
                img = plt.imread(f'./Deck_Icons/{deck}.png')
                st.image(img, use_column_width=True)
            with header_col[1]:
                fig = self.__semi_circle_plot(
                    df.loc[idx_deck, ['Siege', 'Remis', 'Niederlage']].to_list())
                st.pyplot(fig, transparent=True)
                st.caption('Wanderpokal:   ' + ':star:' *
                           int(df.at[idx_deck, 'Meisterschaft']['Win']['Wanderpokal']))
                st.caption('Fun Pokal:     ' + ':star:' *
                           int(df.at[idx_deck, 'Meisterschaft']['Win']['Fun Pokal']))
                st.caption('Lokal Wins:     ' + ':star:' *
                           int(df.at[idx_deck, 'Meisterschaft']['Win']['Local']))
                st.caption('Lokal Top:     ' + ':star:' *
                           int(df.at[idx_deck, 'Meisterschaft']['Top']['Fun Pokal']))
            with header_col[2]:
                st.metric('Matches', int(
                    df.at[idx_deck, 'Matches']), f"{int(df.at[idx_deck, 'Siege']+df.at[idx_deck, 'Remis']+df.at[idx_deck, 'Niederlage'])} Spiele")
                st.metric(f"Aktueller Rang", value=f'{platz}', delta=f'Besser als {percentage}%')
                with st.expander('Turniere'):
                    table = pd.DataFrame(df.at[idx_deck, 'Meisterschaft']).copy()
                    table.reset_index(inplace=True)
                    table.rename(columns={'index':''}, inplace=True)
                    st.table(table)
            # display win/lose plot
            with header_col[3]:
                ds = pd.Series(df.loc[idx_deck, hist_cols+['Elo']].values)
                ds = ds[ds > 0].reset_index(drop=True)
                st.metric('Aktuelle Elo', df.at[idx_deck, 'Elo'], int(df.at[idx_deck, 'Letzte 3 Monate']))
                st.metric(f"Beste Elo", value=f"{int(df.loc[idx_deck, hist_cols+['Elo']].max())}",
                          delta=f"Schlechteste Elo: {int(ds.min())}", delta_color='inverse')
            # disply stats spider plot
            with header_col[4]:
                #st.metric('Jahresänderung', value=int(df.loc[idx_deck, 'Letzte 12 Monate']))
                categories = ['Attack', 'Control', 'Recovery',
                              'Consistensy', 'Combo', 'Resilience']
                fig = self.__make_spider_plot(
                    df.loc[idx_deck, categories].astype(int).to_list())
                st.plotly_chart(fig, theme="streamlit",
                                use_container_width=True)
                
            # second row for displaying elo KPI's like rank, titles and historic elo ratings
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
        # generate plot dataframe with elo, date and mean of the last year
        df_plot = pd.DataFrame(columns=['Elo'])
        df_plot['Elo'] = df.loc[idx, hist_cols[-4:]+['Elo']].values
        df_plot['Datum'] = hist_cols[-4:]+['Elo']
        df_plot = df_plot[df_plot['Elo'] > 0].reset_index(drop=True)
        df_plot['Mittelw.'] = df_plot['Elo'].mean()
        # generate plotly figure and plot as well as layout of the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Datum"], y=df_plot["Elo"],
                      mode='lines+markers', name='Verlauf', line=dict(color="aqua", width=5)))
        fig.add_trace(go.Scatter(x=df_plot["Datum"], y=df_plot["Mittelw."],
                      mode='lines+markers', name='Mittelw.', line=dict(color=self.filed_color, width=3)))
        fig.update_traces(textposition="bottom right")
        fig.update_layout(font_size=15, template='plotly_white', title='Eloverlauf des letzen Jahres', xaxis_title='Datum', yaxis_title='Elo',
                          paper_bgcolor=self.back_color, plot_bgcolor=self.back_color)
        return fig

    def __make_spider_plot(self, stats):
        '''
        Method to mmake a spider plot to show deck stats
        :param stats: list with stats of the current deck
        return fig: plotly figure object
        '''
        # define categories and append first stats element for connected lines in plot
        categories = ['Attack', 'Control', 'Recovery',
                      'Consistensy', 'Combo', 'Resilience', 'Attack']
        stats.append(stats[0])
        # generate plotly figure and define layout
        plot = go.Scatterpolar(r=stats, theta=categories, fill='toself', line=dict(
            color=self.filed_color, width=3))
        fig = go.Figure(data=[plot], layout=go.Layout(
            paper_bgcolor=self.back_color, plot_bgcolor=self.back_color))
        fig.update_layout(font_size=15,
                          polar=dict(radialaxis=dict(
                              visible=False, range=[0, 5])),
                          showlegend=False, title='Deck Eigenschaften',
                          hovermode="x",
                          width=380,
                          height=380)
        return fig

    def __semi_circle_plot(self, val):
        '''
        Method for plotting a semi circle plot to show number of wins and losses
        :param val: list with win/remis/loss standing
        :return fig: matplotlib figure object
        '''
        # define bottom categorie and display it in background color
        val = np.append(val, sum(val))  # 50% blank
        colors = ['green', 'blue', 'red', self.back_color]
        explode = 0.05*np.ones(len(val))
        # genete figure object, plot and set layout
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.pie(val, colors=colors, pctdistance=0.85, explode=explode)
        ax.text(-1.05, 1.1,
                f"N {int(val[2]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=14)
        #ax.text(-0.1, 1.1, f"U {int(val[1]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=19)
        ax.text(
            0.65, 1.1, f"S {int(val[0]/(0.5*np.sum(val)+1e-7)*1000)/10}%", color='white', fontsize=14)
        ax.add_artist(plt.Circle((0, 0), 0.6, color=self.back_color))

        return fig
