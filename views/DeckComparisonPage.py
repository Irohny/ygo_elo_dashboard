import numpy as np
import pandas as pd
import streamlit as st
import VisualizationTools as vit
import utils.date_handling as date_handling
from custom_components.icon_list_markdown import icon_list
import utils.global_variables as gv


def create_deck_summary(deck_id: str, col: st):
    deck = df_decks[df_decks["DeckID"] == deck_id].iloc[0]
    results = st.session_state["tournament_model"].get_tournament_result_of_deck_by_id(
        deck_id
    )
    df_tour = tournament[tournament["DeckID"] == deck_id].reset_index(drop=True)
    # header
    img_col = col.columns(2)
    img_col[0].subheader(deck["Tier"])
    img_col[0].subheader(deck["Type"])
    img_col[0].metric("Aktuelle Elo", int(deck["Elo"]), int(deck["Letzte 3 Monate"]))
    # icon
    vit.load_and_display_image(f"./Deck_Icons/{deck['Deck']}.png", pos=img_col[1])

    # stats
    inside_col = col.columns([1, 1, 1])
    inside_col[0].metric("Matches", f"{deck['Matches']}", f"{deck['Spiele']} Spiele")
    inside_col[1].metric("Gegner Stärke", int(deck["DGP"]))
    inside_col[2].metric(
        "Turniere",
        len(df_tour),
        f"{results['Win']}/{results['Draw']}/{results['Loss']}",
    )
    col.markdown("---")

    # Pokale
    plot_tournament_results(results, col)

    # generate win/lose plot
    segment = col.segmented_control(
        "Eigenschaft_vis",
        ["Gewinrate", "Eigenschaften", "Turniere"],
        key=f"Eigenschaft_{deck['DeckID']}",
        label_visibility="collapsed",
        default="Gewinrate",
    )
    if segment == "Gewinrate":
        plot_gewinnrate(deck, col)
    elif segment == "Eigenschaften":
        plot_attributes(deck, col)
    elif segment == "Turniere":
        plot_turnier_rate(df_tour, results, col)
    else:
        col.error("Kein Attribut ausgewählt.")


def plot_tournament_results(results, col):
    inside_col = col.columns([1, 1, 1])
    inside_col[0].markdown("Wanderpokal")
    icon_list("", "trophy", results["wanderpokal"], inside_col[0])
    # local wins
    inside_col[0].markdown("Local Win")
    icon_list("", "trophy", results["local_wins"], inside_col[0])
    # fun pokale
    inside_col[1].markdown("Fun Pokal")
    icon_list("", "trophy", results["fun_pokal"], inside_col[1])
    # local tops
    inside_col[2].markdown("Local Top")
    icon_list("", "star", results["local_tops"], inside_col[2])


def plot_gewinnrate(deck: dict, col: st):
    values = [deck["Siege"], deck["Remis"], deck["Niederlagen"]]
    fig = vit.plotly_gauge_plot(100 * values[0] / sum(values) // 1)
    col.plotly_chart(
        fig,
        transparent=True,
        use_container_width=True,
        key=f"WinRate{deck_id}",
    )


def plot_attributes(deck: dict, col: st):
    categories = [int(deck[attr]) for attr in gv.DECK_ATTRIBUTES]
    fig = vit.make_spider_plot(categories)
    col.plotly_chart(fig, transparent=True, use_container_width=True)


def plot_turnier_rate(df_tour: pd.DataFrame, results: dict, col: st):
    if df_tour.empty:
        tabs[2].error("Keine Turnierergebnisse.")
    else:
        win_rate = int(
            100
            * results["Win"]
            / sum([results["Win"], results["Draw"], results["Loss"]])
        )
        fig = vit.plotly_gauge_plot(win_rate)
        col.plotly_chart(fig, transparent=True, use_container_width=True)


#############################################################
# Page Layout
#############################################################
DECKS_PER_ROW = 4
# build page
df = st.session_state["deck_data"].copy()
tournament = st.session_state["tournament_model"].df.copy()

st.title(":trophy: Deckvergleich :trophy:", anchor="anchor_tag")
# generate a Multiselect field for choosing the decks that elo stats should displayed
decks = st.multiselect(
    "Wähle deine Decks",
    options=list(df["Deck"].unique()),
    default=st.session_state["deck"],
)

if not decks:
    st.stop()
df_decks = df[df["Deck"].isin(decks)].reset_index(drop=True)
decks_plot = []
for i, deck in df_decks.iterrows():
    tmp = st.session_state["history_model"].get_elo_history_of_deck_by_id(
        deck["DeckID"]
    )
    tmp.loc[len(tmp), ["Datum", "Elo"]] = (
        date_handling.get_current_date_as_integer(),
        deck["Elo"],
    )
    tmp["Deck"] = deck["Deck"]
    decks_plot.append(tmp)
decks_plot = pd.concat(decks_plot, ignore_index=True)

# build a plotly figure with past elo data for choosen decks
fig = vit.deck_lineplot(decks_plot)
st.plotly_chart(fig, use_container_width=True, theme=None, transparent=True)

# statistic comparision
n_rows = int(np.ceil(len(decks) / DECKS_PER_ROW))

# loop over all rows
for n_row in range(n_rows):
    # two columns per deck stats
    columns = st.columns(DECKS_PER_ROW)
    for j in range(DECKS_PER_ROW):
        idx_deck = int(n_row * DECKS_PER_ROW + j)
        if idx_deck >= len(decks):
            continue
        # get deck data and combine data
        deck = decks[idx_deck]
        deck_id = df[df["Deck"] == deck]["DeckID"].iloc[0]

        act_col = columns[j].container(border=True)
        act_col.header(deck)

        create_deck_summary(deck_id, act_col)
