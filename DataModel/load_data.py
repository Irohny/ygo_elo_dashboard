import streamlit as st

from .HistoryModel import HistoryModel
from .EloModel import EloModel
from .TournamentModel import TournamentModel
from .YgoCradModel import YgoCardModel

def load_data():
    if "login" not in st.session_state:
        st.session_state["login"] = True
        st.session_state["user_roles"] = ["admin"]
        st.session_state["user_name"] = "Admin"
        st.session_state["deck"] = []
    st.cache_data.clear()
    dm_elo = EloModel()

    # load data models
    st.session_state["elo_model"] = dm_elo
    st.session_state["history_model"] = HistoryModel()
    st.session_state["tournament_model"] = TournamentModel()
    st.session_state["ygo_pro_model"] = YgoCardModel()

    # setup data for dashboard
    paused_decks = st.session_state["history_model"].get_paused_deckIDs()
    paused_decks.set_index("DeckID", inplace=True)

    df = dm_elo.get_all_decks()
    df.set_index("DeckID", inplace=True)
    df["Active"] = True
    df.loc[paused_decks.index, "Active"] = False
    df.reset_index(drop=False, inplace=True)

    elo_changes = st.session_state["history_model"].elo_changes()
    df = df.merge(elo_changes, on="DeckID", how="left").fillna(0)
    df.rename(
        columns={
            "elo_change_3_months": "Letzte 3 Monate",
            "elo_change_6_months": "Letzte 6 Monate",
            "elo_change_12_months": "Letzte 12 Monate",
        },
        inplace=True,
    )
    df.sort_values(by="Elo", ascending=False, inplace=True, ignore_index=True)
    df["EloPlatz"] = df.index.to_numpy() + 1

    df["Spiele"] = df["Siege"] + df["Remis"] + df["Niederlagen"]
    df["Gewinnrate"] = (df["Siege"] / df["Spiele"]).fillna(0)

    st.session_state["deck_data"] = df

    st.session_state["Deck_Names"] = list(df["Deck"].unique())
    st.session_state["Player_Names"] = list(df["Player"].unique())
