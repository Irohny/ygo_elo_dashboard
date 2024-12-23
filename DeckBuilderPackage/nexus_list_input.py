import streamlit as st
import pandas as pd
import re

from DeckBuilderPackage.utils import merge_sst, reset_session_state_for_decks


def nexus_list_input(st_obj: st, feat_cols, ygo_db):
    """
    Method for creating and handling the list input for easy deck tagging
    :param st_obj: streamlit object for placing elements
    """
    # define list input layout with 3 tabs for each deck type
    input_form = st_obj.form("Input")
    input_tabs = input_form.tabs(["Main-Deck", "Extra-Deck", "Side-Deck"])
    placeholder = """1x Left Arm of the Forbidden One"""
    main_str = input_tabs[0].text_area(
        "Main-Deck Liste:", height=480, placeholder=placeholder
    )
    extra_str = input_tabs[1].text_area(
        "Extra-Deck Liste:", height=480, placeholder=placeholder
    )
    side_str = input_tabs[2].text_area(
        "Side-Deck Liste:", height=480, placeholder=placeholder
    )
    # handling data extracktion if input is submitted
    if input_form.form_submit_button("Übertrage Decks"):
        # main
        df_main = create_table_from_decklist(main_str, feat_cols, ygo_db)
        # extra
        df_extra = create_table_from_decklist(extra_str, feat_cols, ygo_db)
        # side
        df_side = create_table_from_decklist(side_str, feat_cols, ygo_db)
        # merge decks into session state
        merge_into_session_state(df_main, df_side, df_extra, feat_cols)


def create_table_from_decklist(string, feat_cols, ygo_db):
    """
    Method for creating a dataframe with all feature of list input data
    :param string: input string with list of cards
    :return datframe: dataframe with card feature
    """
    # return empty dataframe if list is empty
    if not string:
        return pd.DataFrame(columns=feat_cols)
    # find patterns and create dataframe
    matches = split_text_by_pattern(string)
    df = create_dataframe(matches)
    # merge ygo database data to extracted cards
    return pd.merge(df, ygo_db, left_on="Karte", right_on="name", how="left")


def merge_into_session_state(df_main, df_side, df_extra, feat_cols):
    """
    Method for adding/merging new cards to deck dataframe
    :param df_main: ain decj dataframe
    :param df_side: side deck dataframe
    :param df_extra: extra deck dataframe
    """
    # init session state if not existing
    if "main_deck" not in st.session_state:
        reset_session_state_for_decks(feat_cols)
    # merge dataframes
    if not df_main.empty:
        st.session_state["main_deck"] = merge_sst(
            st.session_state["main_deck"], df_main
        )
    if not df_side.empty:
        st.session_state["side_deck"] = merge_sst(
            st.session_state["side_deck"], df_side
        )
    if not df_extra.empty:
        st.session_state["extra_deck"] = merge_sst(
            st.session_state["extra_deck"], df_extra
        )


def split_text_by_pattern(text: str) -> list:
    """
    Method for extracting cards and amount of a big string
    Pattern 1x Stovie 2x Arianna
    :param text: string of all cards matching the pattern
    :return: list of matches patterns
    """
    # Verwendet regulären Ausdruck, um nach dem Muster "Number x Name" zu suchen
    pattern = re.compile(r"(\b\d+x+\s\b[a-zA-Z0-9,.:&\-' ]+)")
    # Findet alle Übereinstimmungen im Text
    return pattern.findall(text)


def create_dataframe(card_list: list) -> pd.DataFrame:
    """
    Method for creating a dataframe with all crads from the list input
    matched patterns
    :param card_list: list of strings with matched patterns of list input
    :return df: datframe with extracted feature of list input
    """
    # init features
    df = pd.DataFrame(columns=["Karte", "Anzahl", "Tag"])
    cards = []
    amounts = []
    # loop through data and extract need feature
    for card in card_list:
        splits = card.split("x ")
        amounts.append(int(splits[0]))
        cards.append("".join(splits[1:]).strip())
    # set feature to dataframe
    df["Karte"] = cards
    df["Anzahl"] = amounts
    df["Tag"] = "Rest"
    return df
