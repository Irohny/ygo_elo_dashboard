import streamlit as st
import numpy as np
import pandas as pd

import VisualizationTools as vit
from custom_components.icon_list_markdown import icon_list
import utils.global_variables as gv


def display_player_stats(name: str, frame: str, st_obj: st) -> None:
    """
    Method for creating a player stats aggregation
    :param name: name of the player
    :param df: dataframe with decks
    :param hist_cols: history columns of df
    :param st_obj: streamlit object for layout
    """
    df_player = df[df["Player"] == name].reset_index(drop=True)
    df_tour = tournament[
        tournament["DeckID"].isin(df_player["DeckID"].values)
    ].reset_index(drop=True)
    # display player stats
    st_obj.header(name)

    pokale = [
        st.session_state["tournament_model"].get_tournament_result_of_deck_by_id(
            deck_id
        )
        for deck_id in df_player["DeckID"].values
    ]
    pokale = sum_dictionaries(pokale)

    if frame == "Statistik":
        subcols = st_obj.columns(2)
        best_elo = st.session_state["elo_model"].get_best_elo_by_player(name)
        best_hist = st.session_state["history_model"].get_best_elo_by_player(name)

        subcols[0].metric(
            "Decks", len(df_player), delta=f"Aktiv {sum(df_player['Active'])}"
        )
        subcols[0].metric(
            "Beste Elo",
            value=f"{best_elo.at[0, 'MaxElo']}",
            delta=f"Insgesamt {best_hist.at[0, 'MaxElo']}",
            delta_color="off",
        )
        games = (
            df_player["Siege"].sum()
            + df_player["Remis"].sum()
            + df_player["Niederlagen"].sum()
        )
        subcols[1].metric(
            "Matches", f"{int(df_player['Matches'].sum())}", f"{games} Spiele"
        )
        subcols[1].metric(
            "Schlechteste Elo",
            value=f"{best_elo.at[0, 'MinElo']}",
            delta=f"Insgesamt {best_hist.at[0, 'MinElo']}",
            delta_color="off",
        )

        icon_list("Wanderpokal ", "star", int(pokale["wanderpokal"]), st_obj)
        icon_list("Fun Pokal   ", "star", int(pokale["fun_pokal"]), st_obj)
        icon_list("Local Win   ", "star", int(pokale["local_wins"]), st_obj)
        icon_list("Local Top   ", "star", int(pokale["local_tops"]), st_obj)

    elif frame == "Eigenschaften":
        spiderplot = vit.make_spider_plot(
            list(df_player[gv.DECK_ATTRIBUTES].mean().astype(int).values)
        )
        st_obj.plotly_chart(spiderplot, use_container_width=True)

    elif frame == "Gewinrate":
        tmp = df_player[["Siege", "Remis", "Niederlagen"]].sum().to_list()
        win_rate = vit.plotly_gauge_plot(100 * tmp[0] / sum(tmp) // 1)
        st_obj.plotly_chart(win_rate, use_container_width=True)

    elif frame == "Tiers":
        tmp = df_player.groupby("Tier")["Deck"].count().reset_index()
        tmp = order_strings(tmp, "Tier", gv.ELO_TIERS)
        tier_plot = vit.ploty_bar(tmp, "Tier", "Deck", False)
        st_obj.plotly_chart(tier_plot, use_container_width=True)

    elif frame == "Decktypen":
        tmp = df_player.groupby("Type")["Deck"].count().reset_index()
        categorie_histogram = vit.ploty_bar(tmp, "Type", "Deck", False)
        st_obj.plotly_chart(categorie_histogram, use_container_width=True)

    elif frame == "Turnier":
        subcols = st_obj.columns(2)
        n_tours = len(df_tour)
        top_rate = (pokale["local_tops"] + pokale["local_wins"]) / n_tours
        subcols[0].metric("Turniere", n_tours)
        subcols[1].metric("Top-Rate", f"{int(100*top_rate)}%")
        tmp = [df_tour["Win"].sum(), df_tour["Draw"].sum(), df_tour["Loss"].sum()]
        win_rate = vit.plotly_gauge_plot(int(100 * tmp[0] / sum(tmp)))
        st_obj.plotly_chart(win_rate, use_container_width=True, key=f"WinRate{name}")


def order_strings(df, col, order):
    res = []
    values = list(df[col].unique())
    for tag in order:
        if tag not in values:
            continue
        idx = df[df[col] == tag].index
        res.append(df.loc[idx])
    return pd.concat(res, ignore_index=True)


def sum_dictionaries(dict_list):
    """
    Summiert die Werte aller Dictionaries in einer Liste.

    :param dict_list: Liste von Dictionaries mit numerischen Werten.
    :return: Ein einzelnes Dictionary mit aufsummierten Werten.
    """
    summed_dict = dict_list[0].copy()

    for d in dict_list[1:]:
        for key, value in d.items():
            summed_dict[key] += value

    return dict(summed_dict)


###########################################################
# Page Build
###########################################################
df = st.session_state["deck_data"].copy()
tournament = st.session_state["tournament_model"].df.copy()

st.title(":trophy: Spieler :trophy:", anchor="anchor_tag")
inputs = st.columns([2, 2, 1])

frame = inputs[0].segmented_control(
    "StatsAuswahl",
    [
        "Statistik",
        "Eigenschaften",
        "Gewinrate",
        "Tiers",
        "Decktypen",
        "Turnier",
    ],
    default="Statistik",
    label_visibility="collapsed",
)

players = inputs[1].multiselect(
    "Spieler:",
    options=list(df["Player"].unique()),
    default=["Christoph", "Frido", "Jan", "Thomas"],
    label_visibility="collapsed",
)
# skip processing if no player is selected
if not players:
    st.stop()

player_cols = st.columns(len(players), border=True)
# Chriss
for idx, player in enumerate(players):
    display_player_stats(player, frame, player_cols[idx])
