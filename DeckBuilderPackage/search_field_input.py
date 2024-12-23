import requests
import re
import numpy as np
import pandas as pd
import streamlit as st

from DataModel import YgoCradModel
from DeckBuilderPackage.utils import merge_sst


def search_field_input(st_obj: st, card_model: YgoCradModel):
    """
    Method for handling the Search field and card selection for input tab
    param st_obj: streamlit position object
    """
    # define search field
    text = st_obj.text_input("Suche Karte:", placeholder="Test")
    df = card_model.fuzzy_search(text)
    # define dataeditor for displaying search results
    form = st_obj.form("searcher")
    df_select = form.data_editor(
        df[["image", "Hinzufügen", "name"]],
        hide_index=True,
        height=500,
        column_config={"image": st.column_config.ImageColumn("", width="small")},
    )
    # submit selected search results to card tagging
    cols = form.columns(3)
    if cols[0].form_submit_button("Zum Main"):
        add_selection_to_session_state(df_select, df, "main")
    if cols[1].form_submit_button("Zum Side"):
        add_selection_to_session_state(df_select, df, "side")
    # if cols[2].form_submit_button('Zum Extra'):
    #    add_selection_to_session_state(df_select, df, 'extra')


def add_selection_to_session_state(
    df_select: pd.DataFrame, df: pd.DataFrame, option: str
):
    """
    Method for adding selected searched cards to tagging and analysis dataframes
    :param df_selected: dataframe with cards for adding
    :param df: dataframe with all seached cards
    :param option: part of deck for adding
    """
    # get all feature of the choosen cards
    idx = df[df_select["Hinzufügen"]].index
    tmp = df.loc[idx].reset_index(drop=True)

    if idx.empty:
        return
    if (
        option == "main"
        and tmp["frameType"].isin({"synchro", "fusion", "xyz", "link"}).any()
    ):
        idx_extra = tmp[
            tmp["frameType"].isin({"synchro", "fusion", "xyz", "link"})
        ].index
        idx_main = tmp[
            ~tmp["frameType"].isin({"synchro", "fusion", "xyz", "link"})
        ].index
        st.session_state["extra_deck"] = merge_sst(
            st.session_state["extra_deck"], tmp.loc[idx_extra]
        )
        tmp = tmp.loc[idx_main]

    # merge selection to session state deck dataframe
    st.session_state[f"{option}_deck"] = merge_sst(
        st.session_state[f"{option}_deck"], tmp
    )
    st.rerun()
