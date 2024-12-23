import numpy as np
import pandas as pd
import streamlit as st

import VisualizationTools as vit
from utils.date_handling import convert_integer_to_date, get_current_date_as_integer
from custom_components.icon_list_markdown import icon_list
import utils.global_variables as gv


def header_row():
    """ """
    st.title(":trophy: Deck Details :trophy:", anchor="anchor_tag")
    # set data format and display a select box for choosing the deck of interest
    deck_ops = df["Deck"].astype(str).to_numpy()
    deck = st.selectbox(
        "DeckChoosing",
        options=deck_ops,
        label_visibility="collapsed",
        placeholder="Wähle ein deck",
    )

    deck_id = df[df["Deck"] == deck]["DeckID"].iloc[0]
    if len(deck) == 0:
        return None, None
    idx_deck = int(df[df["Deck"] == deck].index.to_numpy())
    # tournament data
    tournament = st.session_state["tournament_model"].df
    tournament = tournament[tournament["DeckID"] == deck_id].reset_index(drop=True)
    # deck feature
    deck = df.loc[idx_deck]
    deck["percentage"] = ((len(df) - deck["EloPlatz"] + 1) / len(df) * 100).astype(int)

    return deck, tournament


def metric_block(pos: st, deck, elos, dates, tournament):
    """ """
    dy = np.array(elos)
    dy = dy[dy > 0]
    fig = vit.plot_metric(
        label="Aktuelle Elo",
        value=deck["Elo"],
        x_data=dates,
        y_data=(dy - 0.95 * min(dy)),
        show_graph=True,
        color_graph="rgba(0, 104, 201, 0.2)",
    )
    pos.plotly_chart(fig, use_container_width=True)
    pos.markdown("---")
    inside = pos.columns([1, 1, 1, 1])
    # Metric Block
    inside[0].metric("Aktueller Rang:", value=f"{int(deck['EloPlatz'])}")
    inside[0].metric("Besser als:", f"{deck['percentage']}%")

    inside[1].metric("Matches", int(deck["Matches"]))
    inside[1].metric("Spiele:", f"{int(deck['Spiele'])}")

    inside[2].metric("Beste Elo:", value=f"{int(max(elos))}")
    inside[2].metric("Letzte Änderung:", value=deck["Letzte 3 Monate"])

    inside[3].metric("Turniere", len(tournament))
    win, draw, loss = 0, 0, 0
    if not tournament.empty:
        win = tournament["Win"].sum()
        draw = tournament["Draw"].sum()
        loss = tournament["Loss"].sum()
    inside[3].metric("Turnier Ergebnisse S/U/N", f"{win}/{draw}/{loss}")


def line_plot(deck: dict, elos: list[int], dates: list[int]):
    """
    Method for creating a lineplot of the elo points of the last year
    :param df: dataframe with historic elo
    :param hist_cols: list wit historic columns
    :param idx: index of the current deck
    :return fig: plotly figure object
    """
    # generate plot dataframe with elo, date and mean of the last year
    df_plot = pd.DataFrame(columns=["Elo"])
    df_plot["Elo"] = elos[-6:]
    df_plot["Datum"] = dates[-6:]
    df_plot = df_plot[df_plot["Elo"] > 0].reset_index(drop=True)
    df_plot["Mittelw."] = df_plot["Elo"].mean()
    # generate plotly figure and plot as well as layout of the figure
    return vit.lineplot(df_plot["Datum"], df_plot["Elo"], df_plot["Mittelw."])


#########################################
# build layout
#########################################
df = st.session_state["deck_data"].copy()
deck, tournament = header_row()
# if deck was choosen display the stats
if deck is None:
    st.stop()

header_col = st.columns([2, 4])
# Layout of deck KPI's, organisation in two rows
# first row with deck name, tier, Icon, type, win/lose plot and stats spider
# display deck name, icon and type
head0 = header_col[0].container(border=False).columns(2)
vit.load_and_display_image(
    f'./Deck_Icons/{deck["Deck"]}.png',
    f'{deck["Tier"]},    {deck["Type"]}',
    pos=head0[0],
)
if not deck["Active"]:
    head0[0].error("Deck ist pausiert")
# win rate plot
vit.winrate_widget(head0[1], deck)

# stats part
hist_elo = st.session_state["history_model"].get_elo_history_of_deck_by_id(
    deck["DeckID"]
)
hist_elo.sort_values(by="Datum", ascending=True, inplace=True, ignore_index=True)
elos = hist_elo["Elo"].to_list()
elos.append(deck["Elo"])
dates = hist_elo["Datum"].to_list()
dates.append(get_current_date_as_integer())
dates = [convert_integer_to_date(d) for d in dates]
center_col = header_col[1].container(border=True)
metric_block(center_col, deck, elos, dates, tournament)

# disply stats spider plot
fig = vit.make_spider_plot([int(deck[c]) for c in gv.DECK_ATTRIBUTES])
head0[0].plotly_chart(fig, use_container_width=True)

# Win stats
results = st.session_state["tournament_model"].get_tournament_result_of_deck_by_id(
    deck["DeckID"]
)
icon_list("Wanderpokal", "trophy", results["wanderpokal"], head0[1])
icon_list("Fun Pokal  ", "star", results["fun_pokal"], head0[1])
icon_list("Local Win  ", "medal", results["local_wins"], head0[1])
icon_list("Local Top  ", "star", results["local_tops"], head0[1])

plot = header_col[1].segmented_control(
    "PlotAuswahl",
    ["Eloverlauf", "Turniere"],
    default="Eloverlauf",
    label_visibility="collapsed",
)
if plot == "Eloverlauf":
    fig = line_plot(deck, elos, dates)
else:
    tournament.sort_values("Date", inplace=True, ignore_index=True)
    fig = vit.tournament_date_bar_plot(tournament)
if fig is not None:
    header_col[1].plotly_chart(fig, theme=None, use_container_width=True)
else:
    header_col[1].error("Es liegen keine Turnierergebnisse vor.")
