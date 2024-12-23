import pandas as pd
import streamlit as st


def merge_sst(df1, df2):
    """
    Add Cards to datafme that are not in the dataframe
    :param df1: dataframe into merging new cards
    :param df2: dataframe with cards for merging
    """
    idx = df2[~df2["Karte"].isin(df1["Karte"].to_list())].index
    cols = list(df1.columns)
    return df1 if idx.empty else pd.concat([df1, df2.loc[idx, cols]], ignore_index=True)


def reset_session_state_for_decks(feat_cols):
    """
    Method for intialise or reset session state deck data
    """
    st.session_state["main_deck"] = pd.DataFrame(columns=feat_cols)
    st.session_state["side_deck"] = pd.DataFrame(columns=feat_cols)
    st.session_state["extra_deck"] = pd.DataFrame(columns=feat_cols)
