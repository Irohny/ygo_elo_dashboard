import streamlit as st
import pandas as pd

from DataModel.DeckModel import DeckModel
from DeckBuilderPackage.utils import reset_session_state_for_decks
from Authentificator.check_roles import is_player


def database_handler(
    st_obj: st, df_elo: pd.DataFrame, ygo_db: pd.DataFrame, feat_cols: list
):
    """
    Method for handling the deck with databse options
    """
    if not proof_login_status(st_obj):
        return

    deck_options = get_possible_decks(df_elo)
    if deck_options is None:
        st_obj.error("Du hast keine Decks in der Wertung.")
        return

    selected_deck = st_obj.selectbox("Deckauswahl", deck_options)
    if selected_deck == "Keine Auswahl getroffen":
        return

    cols = st_obj.columns(4)
    cols[0].button(
        "Lade Deckliste", on_click=load_decklist, args=[selected_deck, ygo_db]
    )
    cols[1].button("Speicher Deckliste", on_click=save_decklist, args=([selected_deck]))
    cols[2].button("Leere Decklisten", on_click=clear_decklists, args=([feat_cols]))
    cols[3].button("Lösche Deckliste", on_click=delete_decklist, args=([selected_deck]))


def proof_login_status(st_obj: st):
    """
    Method for logging tracking
    """
    if not st.session_state["login"]:
        st_obj.error("Bitte melde dich an um Decks zu laden oder zu speichern")
        return False
    if not is_player(st.session_state["user_roles"]):
        st_obj.error(
            "Du hast keinen Zugriff auf Decklisten. Bitte Melde dich beim Admin um die Berechtigungen zu erhalten."
        )
        return False
    return True


def get_possible_decks(df_elo: pd.DataFrame) -> list:
    """
    Method for getting the possible deck options for seletion
    """
    options = ["Keine Auswahl getroffen"]
    if st.session_state["user_name"] == "Admin":
        opts = list(df_elo["Deck"].unique())
    else:
        opts = list(
            df_elo[df_elo["Owner"] == st.session_state["user_name"]]["Deck"].unique()
        )
    if len(opts) == 0:
        return None
    options.extend(opts)
    return options


def load_decklist(deck, ygo_db):
    """
    Method for loading the decklist from database
    """
    print(deck)
    DeckModel().load_deck(str(deck), st.session_state["user_name"], ygo_db)


def save_decklist(deck):
    """ """
    DeckModel().update_deck(str(deck), st.session_state["user_name"])


def delete_decklist(deck: pd.DataFrame):
    """ """
    DeckModel.remove_from_db(deck)


def clear_decklists(cols):
    """ """
    reset_session_state_for_decks(cols)
