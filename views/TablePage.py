import numpy as np
import pandas as pd
import streamlit as st

import utils.global_variables as gv


def color_arrow(val):
    return "color: green" if val > 0 else "color: red" if val < 0 else "color: white"


def format_arrow(val):
    return f"{'↑' if val > 0 else '↓'} {abs(val):.0f}" if val != 0 else f"{val:.0f}"


def create_popover_filter(col=st) -> dict:
    """ """
    form = col.popover(":mag: Filter")
    # form = expander.form("table_filter")

    # 1. slider for elo range
    slider_range = form.slider(
        "Stellle den Elo-Bereich ein:",
        min_value=int(df["Elo"].min() / 10) * 10,
        max_value=int(np.ceil(df["Elo"].max() / 10)) * 10,
        value=[
            int(df["Elo"].min() / 10) * 10,
            int(np.ceil(df["Elo"].max() / 10)) * 10,
        ],
        step=10,
    )

    cols = form.columns([1, 1, 1])
    return {
        "Player": selection_checkbox(
            st.session_state["Player_Names"], "Spieler", cols[0]
        ),
        "Types": selection_checkbox(gv.DECK_PLAYSTYLS, "Decktypen", cols[1]),
        "Tier": selection_checkbox(gv.ELO_TIERS, "Deck-Tiers", cols[2]),
        "Elo-Range": slider_range,
    }

    # form.form_submit_button("Aktualisiere Tabelle")


def use_filter_for_table(param: dict) -> pd.DataFrame:
    df_select = df.copy()
    if param["active"]:
        df_select = df_select[df_select["Active"]]
    # filter for owner tier and type
    df_select = df_select[df_select["Player"].isin(param["Player"])]
    df_select = df_select[df_select["Tier"].isin(param["Tier"])]
    df_select = df_select[df_select["Type"].isin(param["Types"])]
    df_select = df_select[
        (df_select["Elo"] >= param["Elo-Range"][0])
        & (df_select["Elo"] <= param["Elo-Range"][1])
    ]
    df_select = df_select.reset_index()

    df_select = df_select.sort_values(by=["Elo"], ascending=False).reset_index(
        drop=True
    )
    df_select["Platz"] = df_select.index.to_numpy() + 1
    df_select["Gewinnrate"] = np.round(df_select["Gewinnrate"], 2)
    return df_select


def selection_checkbox(list_data: list, title: str, col: st):
    col.caption(title)
    res = []
    list_data = np.array(list_data)
    for i, data in enumerate(list_data):
        res.append(col.checkbox(data, value=True))
    return list_data[res]


########################################
# Page building
########################################
df = st.session_state["deck_data"].copy()
tournament_cols = [
    "Platz",
    "Deck",
    "Tourn_Win",
    "Tourn_Loss",
    "Tourn_Draw",
    "Turniere",
    "Top-Rate",
    "Match-Win-Rate",
    "Points",
]
deck_cols = [
    "Platz",
    "Deck",
    "Elo",
    "Gewinnrate",
    "Letzte 3 Monate",
    "Letzte 6 Monate",
    "Letzte 12 Monate",
    "Matches",
    "Siege",
    "Remis",
    "Niederlagen",
]
tournament = st.session_state["tournament_model"].df.copy()

st.title(":trophy: Ewige Tabelle :trophy:", anchor="table")
cols = st.columns([1, 1, 1, 3], vertical_alignment="bottom")
table_categorie = cols[0].segmented_control(
    "Wertung", ["Elo", "Turnierscore"], default="Elo"
)

filter_parametr = create_popover_filter(cols[2])
filter_parametr["active"] = cols[1].toggle("Aktive Decks", value=False)
df_select = use_filter_for_table(filter_parametr)

# Setup selected data
if table_categorie == "Elo":
    #
    vis_columns = deck_cols
    data = df_select[deck_cols]
    subset = ["Letzte 3 Monate", "Letzte 6 Monate", "Letzte 12 Monate"]
    data[subset] = data[subset].astype(int)
    data["Gewinnrate"] = (100 * data["Gewinnrate"]).astype(int)
    data = data[vis_columns]
    data = data.style.format(format_arrow, subset=subset).applymap(
        color_arrow, subset=subset
    )
    config = {
        "Platz": st.column_config.NumberColumn(width="small"),
        "ELO": st.column_config.NumberColumn(width="small"),
        "Letzte 3 Monate": st.column_config.NumberColumn(
            "Letzte Änderung", width="small"
        ),
        "Gewinnrate": st.column_config.NumberColumn(width="small"),
    }
else:
    vis_columns = tournament_cols
    data = st.session_state["tournament_model"].df_score
    data = data[data["DeckID"].isin(df_select["DeckID"])]
    df_select = df_select[df_select["DeckID"].isin(data["DeckID"])]
    data.set_index("DeckID", inplace=True)
    df_select.set_index("DeckID", inplace=True)
    data.loc[df_select.index, "Deck"] = df_select["Deck"]
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    data.sort_values("Points", ascending=False, ignore_index=True, inplace=True)
    data["Platz"] = data.index.to_numpy() + 1
    config = {
        "Platz": st.column_config.NumberColumn("Platz", width="small"),
        "Deck": st.column_config.TextColumn("Deck", width="medium"),
        "Tourn_Win": st.column_config.NumberColumn("Siege", width="small"),
        "Tourn_Loss": st.column_config.NumberColumn("Niederlagen", width="small"),
        "Tourn_Draw": st.column_config.NumberColumn("Remis", width="small"),
        "Turniere": st.column_config.NumberColumn("Turniere", width="small"),
        "Top-Rate": st.column_config.NumberColumn("Top-Rate", width="small"),
        "Match-Win-Rate": st.column_config.NumberColumn("Win-Rate", width="small"),
        "Points": st.column_config.NumberColumn("Punkte", width="small"),
    }
    data = data[vis_columns]
st.dataframe(
    data,
    height=820,
    hide_index=True,
    use_container_width=True,
    column_config=config,
)
