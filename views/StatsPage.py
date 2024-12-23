import numpy as np

import streamlit as st
import pandas as pd

import VisualizationTools as vit


def first_kpi_row():
    """
    Method for displaying the first row of the front page KPI's
    Show number of player, games, deck
    """
    n = len(df["Deck"].unique())

    cols = st.columns(3)
    # first box with number of decks and matches
    total_box = cols[0].container(border=True).columns(2, vertical_alignment="bottom")
    total_box[0].metric(":flower_playing_cards: Decks in Wertung:", n)
    total_box[1].metric(":flower_playing_cards: Aktive Decks:", sum(df["Active"]))

    # second box with number of games and player
    player_box = cols[1].container(border=True).columns(3)
    player_box[0].metric(
        ":crossed_swords: Gesamtzahl Matches:", int(df["Matches"].sum() / 2)
    )
    player_box[1].metric(
        ":crossed_swords: Gesamtzahl Spiele:",
        int((df["Siege"].sum() + df["Remis"].sum() + df["Niederlagen"].sum()) / 2),
    )
    player_box[2].metric(
        ":bust_in_silhouette: Anzahl Spieler:", len(df["Player"].unique())
    )
    # third box with number of tournaments and top rating
    tournament_box = cols[2].container(border=True).columns(2)
    tournament = st.session_state["tournament_model"].df
    if tournament is not None:
        tournament_box[0].metric(":bank: Turnierteilnahmen", len(tournament))
        tournament_box[1].metric(
            ":medal: Top-Rate",
            f'{np.round(len(tournament[tournament["Result"]=="Top"])/len(tournament)*100,2)}%',
        )


def second_kpi_row():
    """
    Method for displaying the second KPI row of the dashboard frontpage
    """
    # get best and worst decks in the data
    # define layout columns with header for the following icons and metrics
    cols = st.columns([1, 1, 1], vertical_alignment="top")
    second_row_left(cols[0])
    second_row_middle(cols[1])
    second_row_right(cols[2])


def second_row_left(col0: st):
    # first column: all over best elo,
    selection = col0.segmented_control(
        "ScoreControl",
        ["Turnierscore", "Beste Elo", "Schleteste Elo"],
        label_visibility="collapsed",
        default="Turnierscore",
        selection_mode="single",
    )
    if selection in ["Beste Elo", "Schleteste Elo"]:
        deck_elo_widget(selection, col0)
    else:
        tour_local = tour.merge(df[["DeckID", "Deck", "Elo"]], on="DeckID")
        row = tour_local[tour_local["Points"] == tour_local["Points"]].iloc[0]
        c = col0.columns(2)
        vit.load_and_display_image(
            pos=c[0],
            path=f'./Deck_Icons/{row["Deck"]}.png',
            subtext=f'{row["Deck"]} {row["Elo"]}',
        )
        tmp = (
            tour_local[tour_local["Platz"] > 0]
            .copy()
            .sort_values("Platz", ignore_index=True)
        )
        tmp.rename(columns={"Points": "Punkte"}, inplace=True)
        c[1].dataframe(tmp[["Platz", "Deck", "Punkte"]].head(10), hide_index=True)


def deck_elo_widget(header: str, col: st):
    """ """
    hist_stats = st.session_state["history_model"].get_history_statistics()
    if header == "Beste Elo":
        hist_stats_max = hist_stats[hist_stats["Elo"] == hist_stats["Elo"].max()].iloc[
            0
        ][["DeckID", "Elo"]]
        elo_stats = df[df["Elo"] == df["Elo"].max()].iloc[0][["DeckID", "Elo", "Deck"]]

        stats = elo_stats
        if hist_stats_max["Elo"] > elo_stats["Elo"]:
            stats = hist_stats_max

        deck_actuel = elo_stats["Deck"]
        elo_actuel = elo_stats["Elo"]

        tmp = df[df["DeckID"] == elo_stats["DeckID"]].reset_index(drop=True)
        deck_ever = tmp.at[0, "Deck"]
        elo_ever = int(tmp.at[0, "Elo"])

    elif header == "Schleteste Elo":
        hist_stats_min = hist_stats[hist_stats["Elo"] == hist_stats["Elo"].min()].iloc[
            0
        ][["DeckID", "Elo"]]
        elo_stats = df[df["Elo"] == df["Elo"].min()].iloc[0][["DeckID", "Elo", "Deck"]]

        stats = elo_stats
        if hist_stats_min["Elo"] < elo_stats["Elo"]:
            stats = hist_stats_min

        deck_actuel = elo_stats["Deck"]
        elo_actuel = elo_stats["Elo"]

        tmp = df[df["DeckID"] == elo_stats["DeckID"]].reset_index(drop=True)
        deck_ever = tmp.at[0, "Deck"]
        elo_ever = int(tmp.at[0, "Elo"])

    c = col.columns(2)
    vit.load_and_display_image(
        pos=c[0],
        title="Aktuell",
        path=f"./Deck_Icons/{deck_actuel}.png",
        subtext=f"{deck_actuel} {elo_actuel}",
    )
    vit.load_and_display_image(
        pos=c[1],
        title="Total",
        path=f"./Deck_Icons/{deck_ever}.png",
        subtext=f"{deck_ever} {elo_ever}",
    )


def second_row_middle(col=st):
    """ """
    selection = col.segmented_control(
        "PlotControll",
        ["Decktypen", "Elo", "Tiers"],  # "Zeitstrahl",
        default="Decktypen",
        label_visibility="collapsed",
    )
    if selection == "Elo":
        fig = vit.box_violin_plot(df, "Type", "Elo", "Eloverteilung pro Decktyp", "Elo")
        col.pyplot(fig, transparent=True)
    elif selection == "Decktypen":
        tmp = df[["Type", "Deck"]].groupby("Type").count().reset_index()
        fig = vit.ploty_bar(
            tmp, "Type", "Deck", False, title="Verteilung der Decks auf die Typen"
        )
        col.plotly_chart(fig, tranparent=True)
    # elif selection == "Zeitstrahl":
    #    with col:
    #        vit.timeline_for_decks()
    elif selection == "Tiers":
        fig = vit.deck_density(np.array(df["Elo"].values).squeeze())
        col.pyplot(fig, transparent=True)


def second_row_right(col: st):
    """ """
    selection = col.segmented_control(
        "TypeControl",
        list(df["Type"].unique()),
        default="Midrange",
        label_visibility="collapsed",
    )

    cols = col.columns([0.5, 1, 0.5])
    tmp = df[df["Type"] == selection].copy().reset_index(drop=True)
    idx = tmp["Elo"].argmax()
    vit.load_and_display_image(
        pos=cols[1],
        path=f'./Deck_Icons/{tmp.at[idx, "Deck"]}.png',
        subtext=f"{tmp.at[idx, 'Deck']} {tmp.at[idx, 'Elo']}",
    )
    n = len(tmp)
    cols[1].markdown(
        f"<h5 style='text-align: center; color: withe;' >Decks {n}</h5>",
        unsafe_allow_html=True,
    )


#########################################
# Page Building
#########################################
df = st.session_state["deck_data"].copy()
tour = st.session_state["tournament_model"].df_score.copy()

st.title(":trophy: YuGiOh! Elo-Dashboard :trophy:", anchor="anchor_tag")
first_kpi_row()
second_kpi_row()
