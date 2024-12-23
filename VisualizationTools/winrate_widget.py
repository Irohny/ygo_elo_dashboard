import streamlit as st
from .Visualization import plotly_gauge_plot


def winrate_widget(col: st, deck: dict):
    """
    Function for generating a widget that show the win rate of a deck in elo and
    tournament modus
    :param col: streamlit position argument
    :param deck: row in dict format with data of the deck to plot
    """
    win_rate_type = col.segmented_control(
        "WinRateType", ["Elo", "Turnier"], default="Elo", label_visibility="collapsed"
    )
    # get data for plotting
    if win_rate_type == "Elo":
        values = [deck["Siege"], deck["Remis"], deck["Niederlagen"]]
    elif win_rate_type == "Turnier":
        tmp = st.session_state["tournament_model"].df.copy()
        tmp = tmp[tmp["DeckID"] == deck["DeckID"]]
        values = []
        if not tmp.empty:
            values = [tmp["Win"].sum(), tmp["Draw"].sum(), tmp["Loss"].sum()]

    # plotting if data is not empty
    if len(values) > 0:
        fig = plotly_gauge_plot(100 * values[0] / sum(values) // 1)
        col.plotly_chart(fig, use_container_width=True)
    else:
        col.error("Keine Turnier mit dem Deck gefunden!")
