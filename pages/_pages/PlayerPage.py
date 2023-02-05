import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from emoji import emojize
import plotly.graph_objects as go

class PlayerPage:
    '''
    Class for displaying player stats
    '''
    def __init__(self, df, hist_cols):
        self.back_color = "#00172B"
        self.filed_color = "#0083B8"
        self.__build_page_layout(df, hist_cols)

    @st.cache(suppress_st_warning=True)
    def __build_page_layout(self, df, hist_cols):
        '''
        '''
        player_cols = st.columns(5)
        # Chriss
        with player_cols[0]:
            self.__display_player_stats('Christoph', df, hist_cols)                
        # Finn
        with player_cols[1]:
            self.__display_player_stats('Finn', df, hist_cols)        
        # Frido
        with player_cols[2]:
            self.__display_player_stats('Frido', df, hist_cols)
        # Jan
        with player_cols[3]:
            self.__display_player_stats('Jan', df, hist_cols)                   
        # Thomas
        with player_cols[4]:
            self.__display_player_stats('Thomas', df, hist_cols)

    def __display_player_stats(self, name, df, hist_cols):
        '''
        '''
        n_wp, n_decks, n_fun, fig1, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min = self.__get_player_infos(df, name, hist_cols)
        st.header(name)
        st.text(f'Decks in Wertung: {n_decks}')
        tmp = df[df['Owner']==name][['Siege', 'Remis', 'Niederlage']].sum()
        fig = self.__semi_circle_plot(tmp.values)
        st.text(f"Matches: {int(df[df['Owner']==name]['Matches'].sum()//2)}")
        st.text(f"Spiele: {int(tmp.sum()//2)}")
        st.metric('Beste Elo Insgesamt/Aktuell', value=f'{hist_elo_best} / {act_elo_best}', delta=int(act_elo_best-hist_elo_best))
        st.metric('Schlechteste Elo Insgesamt/Aktuell', value=f'{hist_elo_min} / {act_elo_min}', delta=int(act_elo_min-hist_elo_min))
        st.text(f'Wanderpokale: '+ emojize((':star:'*int(n_wp))))
        st.text(f'Fun Pokale: '+ emojize((':star:'*int(n_fun))))
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        st.pyplot(fig, transparent=True)
        st.pyplot(fig2, transparent=True)
        st.pyplot(fig3, transparent=True)

    def __semi_circle_plot(self, val):
        '''
        Method for creating a semi circle plot for displying the win/remisloss rate
        :param val: array with win/remis/loss values
        :return fig: matplotlib figure object
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
    
    def __get_player_infos(self, df, name, hist_cols):
        '''
        Method for KPI estimation by given player
        :param df: dataframe with all data
        :param name: name of the player
        :param hist_cols: list with historic elo ratings 
        :return:
        '''
        cols_spider = ['Attack', 'Control', 'Recovery', 'Consistensy','Combo', 'Resilience']
        n_wp = df[df['Owner']==name]['Wanderpokal'].sum()
        n_fun = df[df['Owner']==name]['Fun Pokal'].sum()
        act_elo_best = df[df['Owner']==name]['Elo'].astype(int).max()
        act_elo_min = df[df['Owner']==name]['Elo'].astype(int).min()
        
        tmp = df[df['Owner']==name][hist_cols+['Elo']].astype(int).max()
        hist_elo_best = int(np.max(tmp[tmp>0]))
        
        tmp = df[df['Owner']==name][hist_cols+['Elo']].astype(int).min()
        hist_elo_min = int(np.min(tmp[tmp>0]))
        
        n_decks = len(df[df['Owner']==name])
        fig = self.__make_spider_plot(list(np.round(df[df['Owner']==name][cols_spider].mean(), 0).astype(int).values))
        fig2 = self.__make_deck_histo(df[df['Owner']==name])
        fig3 = self.__make_type_histo(df, name)
        return n_wp, n_decks, n_fun, fig, fig2, fig3, act_elo_best, act_elo_min, hist_elo_best, hist_elo_min
    
    
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
                        title='Eigenschaften'
                        )
        return fig

    def __make_deck_histo(self, df):
        '''
        Method for creating a bar plot with number of decks in a tier range
        :param df: dataframe a data
        :return: matplotlib figure object 
        '''
        n1 = len(df[df['Tier']=='Kartenstapel'])
        n2 = len(df[df['Tier']=='Fun'])
        n3 = len(df[df['Tier']=='Good'])
        n4 = len(df[df['Tier']=='Tier 2'])
        n5 = len(df[df['Tier']=='Tier 1'])
        n6 = len(df[df['Tier']=='Tier 0'])
        
        fig, axs = plt.subplots(1, figsize=(9,9))
        axs.bar([1,2,3,4,5,6], [n1, n2, n3, n4, n5, n6], color='gray')
        axs.set_xticks([1,2,3,4,5,6])
        axs.set_xticklabels(['Kartenstapel', 'Fun', 'Good', 'Tier 2', 'Tier 1', 'Tier 0'], rotation=-45, color='white')
        axs.set_ylabel('Anzahl Decks', color='white')
        axs.grid()
        axs.set_title('Verteilung der Decks auf die Spielstärken', color='white')
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(colors='white', which='both')
        return fig

    def __make_type_histo(self, df, name):
        '''
        Method for creating a bar plot with deck types for a given player
        :param df: dataframe with all data
        :param name: name of the player
        :return fig: matplotlib figure object
        '''
        types = df['Type'].unique()
        n_types = len(types)
        fig, axs = plt.subplots(1, figsize=(8,8))
        n_decks = np.zeros(n_types)
        for k in range(n_types):
                n_decks[k] = len(df[(df['Type']==types[k])&(df['Owner']==name)])
        axs.bar(range(n_types), n_decks, color='gray')
        axs.set_xticks(range(n_types))
        axs.set_xticklabels(types, rotation=-45, color='white')
        axs.set_ylabel('Anzahl Decks', color='white')
        axs.grid()
        axs.set_title('Verteilung der Decks auf die Typen', color='white')
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(colors='white', which='both')
        return fig
