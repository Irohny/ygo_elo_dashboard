import streamlit as st
from VisualizationTools.Visualization import Visualization

def winrate_widget(col:st, deck:dict):
    """
    Function for generating a widget that show the win rate of a deck in elo and
    tournament modus
    :param col: streamlit position argument
    :param deck: row in dict format with data of the deck to plot
    """
    win_rate_type = col.radio('WinRateType', ['Elo', 'Turnier'], horizontal=True, label_visibility='collapsed')
    # get data for plotting
    if win_rate_type == 'Elo':
        values = [deck['Siege'], deck['Remis'], deck['Niederlage']]
    elif win_rate_type == 'Turnier':
        tmp = st.session_state['tournament_data'].copy()
        tmp = tmp[tmp['Deck']==deck['Deck']]
        print(tmp)
        values = []
        if not tmp.empty:
            values = [tmp['Win'].sum(), tmp['Draw'].sum(), tmp['Loss'].sum()]
    
    # plotting if data is not empty
    if len(values)>0:
        fig = Visualization().plotly_gauge_plot(100*values[0]/sum(values)//1)
        col.plotly_chart(fig, use_container_width=True)
    else:
        col.error('Keine Turnier mit dem Deck gefunden!')
