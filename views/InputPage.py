import streamlit as st
import time

from utils.check_roles import check_roles
from DataModel.load_data import load_data
import utils.global_variables as gv


def insert_new_game(df_elo):
    """ """
    st.header("Trage neue Spielergebnisse ein: ")
    form = st.form("Input new Games")
    inputs = form.columns(6)
    # deck and result 1
    deck_options = [""]
    deck_options.extend(st.session_state["Deck_Names"])
    inputs[1].selectbox("Wähle Deck", options=deck_options, key="deck1")
    inputs[1].number_input("Score ", min_value=0, max_value=100, step=1, key="erg1")

    # deck and result 2
    inputs[3].selectbox("Wähle Deck", options=deck_options, key="deck2")

    inputs[3].number_input("Score ", min_value=0, max_value=100, step=1, key="erg2")

    form.form_submit_button("Submit", on_click=update_rating)


def update_rating():
    df_elo = st.session_state["deck_data"].copy()
    deck_id1 = df_elo[df_elo["Deck"] == st.session_state["deck1"]].iloc[0]["DeckID"]
    deck_id2 = df_elo[df_elo["Deck"] == st.session_state["deck2"]].iloc[0]["DeckID"]
    st.session_state["elo_model"].update_elo_ratings(
        deck_id1, deck_id2, st.session_state["erg1"], st.session_state["erg2"], df_elo
    )
    st.session_state["deck1"] = ""
    st.session_state["deck2"] = ""
    after_update()


def insert_new_tournament_win(df_elo):
    """ """
    # insert a new tournament win
    st.header("Trage neues Turnier ein:")
    form = st.form("Turniere:")
    inputs = form.columns(6)
    # choose deck 1
    result = {
        "Deck": inputs[1].selectbox(
            "Wähle Deck",
            options=st.session_state["Deck_Names"],
            key="deck tournament",
        )
    }
    result["Win"] = inputs[1].number_input("Wins", min_value=0, step=1)
    # choose tournament
    result["Mode"] = inputs[2].selectbox("Wähle Turnier:", options=gv.TOURNAMENT_TYPES)
    result["Draw"] = inputs[2].number_input("Draws", min_value=0, step=1)
    # give result
    result["Standing"] = inputs[3].selectbox("Ergebnis:", options=gv.TOURNAMENT_RESULTS)
    result["Loss"] = inputs[3].number_input("Losses", min_value=0, step=1)
    # get date
    result["Date"] = inputs[4].text_input(
        "Datum:", max_chars=10, placeholder="1970-01-01"
    )
    if form.form_submit_button("Submit"):
        result["DeckID"] = df_elo[df_elo["Deck"] == result["Deck"]].iloc[0]["DeckID"]
        st.session_state["tournament_model"].insert_tournament(result)
        after_update(inputs[4])


def insert_new_deck():
    """ """
    # insert a new deck
    st.header("Trage neues Deck ein:")
    form = st.form("neues_deck")
    inputs = form.columns(6)
    # deck, player and type
    new_deck = {
        "Elo": 1500,
        "DGP": 1500,
        "Siege": 0,
        "Remis": 0,
        "Niederlagen": 0,
        "Matches": 0,
        "Tier": "Tier 2",
    }
    new_deck["Deck"] = inputs[1].text_input("Names des neuen Decks", key="new deck")
    new_deck["Player"] = inputs[1].text_input("Names des Spielers", key="player")
    new_deck["Type"] = inputs[1].selectbox(
        "Wähle einen Decktype:",
        options=st.session_state["deck_data"]["Type"].unique(),
        key="decktype",
    )
    # attack, control and resiliance
    new_deck["Attack"] = inputs[2].number_input(
        "Attack-Rating", min_value=0, max_value=5, step=1, key="attack"
    )
    new_deck["Control"] = inputs[2].number_input(
        "Control-Rating", min_value=0, max_value=5, step=1, key="control"
    )
    new_deck["Resilience"] = inputs[2].number_input(
        "Resilience-Rating", min_value=0, max_value=5, step=1, key="resilience"
    )

    # recovery, combo and consistency
    new_deck["Recovery"] = inputs[3].number_input(
        "Recovery-Rating", min_value=0, max_value=5, step=1, key="recovery"
    )
    new_deck["Combo"] = inputs[3].number_input(
        "Combo-Rating", min_value=0, max_value=5, step=1, key="combo"
    )
    new_deck["Consistency"] = inputs[3].number_input(
        "Consistency-Rating", min_value=0, max_value=5, step=1, key="consistency"
    )

    if form.form_submit_button("Submit"):
        st.session_state["elo_model"].insert_new_deck(new_deck)
        after_update(inputs[4])


def create_elo_history():
    """ """
    # set current elo in history by actual month and year
    st.header("Update History:")
    form = st.form("history_update")
    inputs = form.columns(6)
    if form.form_submit_button("Submit"):
        st.session_state["history_model"].update_history(st.session_state["deck_data"])
        after_update(inputs[4])


def modify_deck():
    """ """
    # modify deck stats
    st.header("Modfiziere Deckstats:")
    form = st.form("Mod Stats")
    inputs = form.columns(6)
    # choosen deck
    deck_choose = inputs[1].selectbox(
        "Wähle Deck", options=st.session_state["Deck_Names"], key="deck_modify"
    )

    # stats and values
    in_stats = inputs[2].selectbox(
        "Wähle Eigenschaft zum verändern:", options=gv.DECK_ATTRIBUTES
    )
    modif_in = inputs[2].number_input(
        "Rating:", min_value=0, max_value=5, step=1, key="type_modifier"
    )

    # type and value
    new_type = inputs[3].selectbox("Neuer Type:", options=gv.DECK_PLAYSTYLS)

    # submit
    if form.form_submit_button("Submit"):
        deck_id = str(
            st.session_state["deck_data"][
                st.session_state["deck_data"]["Deck"] == deck_choose
            ]["DeckID"].iloc[0]
        )
        st.session_state["elo_model"].update_stats(
            deck_id, in_stats, modif_in, new_type
        )
        after_update(inputs[3])


def after_update(pos: st = None) -> None:
    if pos is None:
        pos = st.container()
    else:
        pos = pos.container()
    pos.success("Update erfolgreich!!")
    load_data()
    pos.success("Daten neu geladen!!")
    time.sleep(1)
    pos.empty()


##############################################################
# Page Layout
##############################################################
if not st.session_state["login"]:
    st.error("Bitte melede dich an um diese Seite zu nutzen!")
    st.stop()
elif not check_roles(["admin", "reporter"]):
    st.error("Du bist nicht berechtigt diese Seite zu nutzen")
    st.stop()

game, tour, deck, history, mod_stats = st.tabs(
    ["Neues Spiel", "Turniere", "Neues Deck", "Update", "Deck modifizieren"]
)
with game:
    insert_new_game(st.session_state["deck_data"])
with tour:
    insert_new_tournament_win(st.session_state["deck_data"])
with deck:
    insert_new_deck()
with history:
    create_elo_history()
with mod_stats:
    modify_deck()
