import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        :param value:
        :param title:
        :return fig:
        """
        fig = go.Figure(go.Indicator(
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
        
        # update figure color layout for costum theme
        #fig.update_layout(width=200,
        #                height=150)
        return fig

    def plot_metric(self, label, value, x_data=[], y_data=[], prefix="", suffix="", show_graph=False, color_graph=""):
        """
        Metric component for timeline plot behind a metric display
        :param label:
        :param value:
        :param x_data:
        :param y_data:
        :param prefix:
        :param suffix:
        :param show_graph:
        :param color_graph:
        """
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
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_data, line_shape='spline',
                      mode='lines+markers', name='Verlauf', line=dict(color="aqua", width=5)))
        fig.add_trace(go.Scatter(x=x_data, y=mean,
                      mode='lines+markers', name='Mittelw.', line=dict(color="blue", width=3)))
        #
        fig.update_traces(textposition="bottom right")
        fig.update_layout(font_size=15, template='plotly_white', title='Eloverlauf des letzen Jahres', xaxis_title='Datum', yaxis_title='Elo',
                          paper_bgcolor=self.bc, plot_bgcolor=self.bc)
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