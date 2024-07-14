import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

class Visualization:
    def __init__(self,):
        # get color theme from streamlit layout
        self.pc = st.get_option('theme.primaryColor')
        self.bc = st.get_option('theme.backgroundColor')
        self.sbc = st.get_option('theme.secondaryBackgroundColor')
        self.tc = st.get_option('theme.textColor')

    def plotly_gauge_plot(self, value, title='Gewinnrate [%]'):
        """
        Gauge plot component
        :param value: values for plotting
        :param title: title of the plot
        :return fig: plotly figure object
        """
        return go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = value,
                                
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': title},
                                gauge = {'axis': {'range': [0, 100]},
                                         'bar': {'color': "black"},
                                        'steps' : [
                                            {'range': [0, 33], 'color': "red"},
                                            {'range': [33, 66], 'color': "orange"},
                                            {'range': [66, 100], 'color':'green'}],
                                        }
                                        ))

    def plot_metric(self, label, value, x_data=None, y_data=None, prefix="", suffix="", show_graph=False, color_graph=""):
        """
        Metric component for timeline plot behind a metric display
        :param label: titel of the figure
        :param value: value for displaying
        :param x_data: x data for background time series
        :param y_data: y data of background time series
        :param prefix: prefix of value
        :param suffix: suffix of value
        :param show_graph: bool for showing the backgroung data
        :param color_graph: color of the background data graph
        """
        if x_data is None:
            x_data = []
        if y_data is None:
            y_data = []
        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                value=value,
                gauge={"axis": {"visible": False}},
                number={
                    "prefix": prefix,
                    "suffix": suffix,
                    "font.size": 28
                },
                title={
                    "text": label,
                    "font": {"size": 28},
                },
            )
        )

        if show_graph:      
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    hoverinfo="skip",
                    line_shape='spline',
                    fill="tozeroy",
                    fillcolor=color_graph,
                    line={
                        "color": color_graph,
                    },
                )
            )

        fig.update_xaxes(visible=False, fixedrange=True)
        fig.update_yaxes(visible=False, fixedrange=True)
        fig.update_layout(
            # paper_bgcolor="lightgrey",
            margin=dict(t=30, b=0),
            showlegend=False,
            height=100,
        )
        return fig
    
    def lineplot(self, x_data, y_data, mean):
        """
        Method for gernating a plotly line plot with mean 
        :param x_data: data for the x axis
        :param y_data: data for the y axis
        :param mean: data for the mean of the  series
        :return fig: plotly figure object
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_data, line_shape='spline',
                      mode='lines+markers', name='Verlauf', line=dict(color="aqua", width=5)))
        fig.add_trace(go.Scatter(x=x_data, y=mean,
                      mode='lines+markers', name='Mittelw.', line=dict(color="blue", width=3)))
        #
        fig.update_traces(textposition="bottom right")
        fig.update_layout(font_size=15, xaxis_title='Datum', yaxis_title='Elo',
                          legend=dict(yanchor="top",
                                        y=1,
                                        xanchor="left",
                                        orientation="h",
                                        x=0))
        #title='Eloverlauf des letzen Jahres', 
        #template='plotly_white',paper_bgcolor=self.bc, plot_bgcolor=self.bc
        return fig
    
    def make_spider_plot(self, stats):
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
            color=self.pc, width=3))
        fig = go.Figure(data=[plot])
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
    
    def ploty_bar(self, df, xfeat, yfeat, horizontal=True, title=None):
        """
        Method for generating a bar plot with plotly
        :param df: dataframe with the plotting data
        :param xfeat: column with class feature
        :param yfeat: column with values
        :param horizontal: bool for plot orientation
        :param title: title of the plot
        :return fig: plotly figure object
        """
        if horizontal:
            x, y, ori = df[yfeat], df[xfeat], 'h'
        else: 
            x, y, ori = df[xfeat], df[yfeat], 'v'

        fig = go.Figure(go.Bar(
            x=x,
            y=y,
            orientation=ori,
            ))
        fig.update_layout(title_text=title)
        return fig
    
    def plotly_pie(self, df:pd.DataFrame, values:str, names:str, color=None, color_dict:dict=None, mid=0.5, title=''):
        """
        Method for a plotly pie cahrt
        :param df: dataframe with ploting data
        :param values: name of the value coluumn
        :param names: name of the group column
        :param color: color for plotting
        :param color_dict: dictionaray of names and colors for plotting
        :param mid: size of the empty space in the plot center
        :param title: title of the plot
        :return fig: plotly figure object
        """
        fig = px.pie(df, values=values, names=names, hole=mid, color=color,
                     color_discrete_map=color_dict)
        fig.update_layout(title_text=title)
        return fig
    
    def box_violin_plot(self, df:pd.DataFrame, 
                        categorie:str, 
                        distribution:str, 
                        title:str = '',
                        ylabel:str=''):
        fig, axs = plt.subplots(1, figsize=(8,5))
        categories = list(df[categorie].unique())
        for idx, tmp_type in enumerate(categories):
            axs.violinplot(df[df[categorie]==tmp_type][distribution], positions=[idx], showmeans=True, showextrema=False, showmedians=False)
            axs.boxplot(df[df[categorie]==tmp_type][distribution], positions=[idx])
        axs.set_title(title, color=self.tc)
        axs.set_ylabel(ylabel, color=self.tc)
        axs.set_xticks(range(len(categories)))
        axs.set_xticklabels(categories, rotation=-45, color=self.tc)
        axs.grid()
        
        axs.spines['bottom'].set_color(self.tc)
        axs.spines['left'].set_color(self.tc)
        axs.tick_params(colors=self.tc, which='both')
        return fig
        
    def stacked_bar(self, values):
        """
        """
        colors = ['green', 'orange', 'red']
        labels = ['Sieg', 'Remis', 'Niederlage']

        # Erstellen des horizontalen Barplots
        fig = go.Figure()

        for i, value in enumerate(values):
            fig.add_trace(go.Bar(
                y=['Values'], 
                x=[value], 
                orientation='h',
                name=labels[i],
                marker=dict(color=colors[i]),
                showlegend=False
            ))

        # Layout anpassen
        fig.update_layout(
            barmode='stack',
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),  # Entfernen der x-Achsen-Beschriftungen und Rasterlinien
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),  # Entfernen der y-Achsen-Beschriftungen und Rasterlinien
            showlegend=False
        )
        return fig
    
    def tournament_date_bar_plot(self, df):
        """
        Method for generating a plotly figure with a bar plot and a line plot
        :param df: DataFrame containing the data
        :return fig: plotly figure object
        """
        # Subplots erstellen
        fig = go.Figure()

        # Funktion zum dynamischen Setzen der Textposition
        def get_text_position(y_value, threshold=5):
            return 'inside' if y_value > threshold else 'outside'

        # Wins hinzufügen
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Win'],
            name='Wins',
            marker_color='green',
        ))

        # Draws hinzufügen
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Draw'],
            name='Draws',
            marker_color='orange',
        ))

        # Losses hinzufügen
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Loss'],
            name='Losses',
            marker_color='red',
            text=df['Mode'] + ' ' +df['Standing'] +'<br>' + df['Date'].astype(str),
            textposition=[get_text_position(value) for value in df['Loss']]
        ))
        
        # Layout aktualisieren
        fig.update_layout(
            barmode='stack',
            title='Ergebnisse und Differenz zwischen Sieg und Niederlage',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="left",
                x=0
            )
        )

        fig.update_xaxes(title_text='Turnier')
        fig.update_yaxes(title_text='Anzahl Matches')
        
        return 
    
    def deck_lineplot(self, decks:pd.DataFrame, hist_cols:str):
        '''
        Method for plotting the elo history of a set of choosen deck
        :param df: dataframe with deck elo ratings
        :param deck: list with choosen decks
        :param hist_cols: list with historic elo columns
        :return fig: plotly figure object
        '''
        # proof if a deck was choosen
        if decks.empty:
            return None
        # generate a date list
        dates = hist_cols+['Elo']
        df_plot = []
        # loop over all choosen decks and get elo of the dates
        for d in decks['Deck'].values:
            tmp = pd.DataFrame(columns=['Deck', 'Elo', 'Date', 'helper'], index=range(len(dates)))
            tmp['Date'] = dates
            tmp['helper'] = np.linspace(0, len(dates)-1, len(dates))
            tmp['Deck'] = d
            tmp['Elo'] = decks[decks['Deck']==d][dates].to_numpy().squeeze()
            df_plot.append(tmp)
        # combine deck dataframe to a big dataframe
        df_plot = pd.concat(df_plot, axis=0)
        # set empty historic elo (value:0) to NaN and drop them
        df_plot[df_plot['Elo']==0]=np.nan
        #df_plot = df_plot.dropna()
        elo_min = df_plot['Elo'].min()-15
        elo_max = df_plot['Elo'].max()+15
        df_plot['Elo'] = df_plot['Elo'].fillna(0)
        df_plot = df_plot.sort_values(by='helper', ascending=True).reset_index(drop=True)
        # generate a plotly figure with elo ratings
        fig = px.line(df_plot, x="Date", y="Elo", color='Deck', line_shape='spline')
        # update figure traces for hover effects and line width
        fig.update_traces(textposition="bottom right", line=dict(width=5), hovertemplate=None)
        # update figure layout
        fig.update_layout(font_size=15, title='Eloverlauf', xaxis_title='Datum', yaxis_title='Elo',
                        hovermode="x unified",yaxis=dict(range=[elo_min, elo_max]))

        return fig