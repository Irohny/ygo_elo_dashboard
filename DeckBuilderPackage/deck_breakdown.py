import streamlit as st
import numpy as np
import pandas as pd

import VisualizationTools as vit


def create_breakdown(tab: st):
    """
    Method for creating the deck breackdown tab of the deck analysis
    :param tab: stramlit pacing object
    """
    # breakdown tab
    breakdown_cols = tab.columns(6)
    # calculate deck part price brackdown as metric
    mainprice = calculate_total_price(st.session_state["main_deck"])
    extraprice = calculate_total_price(st.session_state["extra_deck"])
    sideprice = calculate_total_price(st.session_state["side_deck"])
    breakdown_cols[0].metric(
        "Gesamtpreis:", f"{np.round(mainprice+extraprice+sideprice,2)}€"
    )
    breakdown_cols[1].metric("Preis Main Deck:", f"{mainprice}€")
    breakdown_cols[2].metric("Preis Extra Deck:", f"{extraprice}€")
    breakdown_cols[3].metric("Preis Side Deck:", f"{sideprice}€")
    # plot deck rations
    plot_cols = tab.columns([1, 0.7, 1])
    # Archetype: feature claculation and plotting
    df = st.session_state["main_deck"].groupby("archetype")["Anzahl"].sum()
    if not df.empty:
        df = df.reset_index()
        plot_cols[0].plotly_chart(
            vit.ploty_bar(df, "archetype", "Anzahl", title="Archetypes"),
            use_container_width=True,
        )
    # Card Ratio: feature claculation and plotting
    st.session_state["main_deck"]["frameType"].fillna("effect", inplace=True)
    df = (
        st.session_state["main_deck"].groupby("frameType")["Anzahl"].sum().reset_index()
    )
    df = reset_values(df, "frameType", "effect", "Monster")
    df = reset_values(df, "frameType", "spell", "Zauber")
    df = reset_values(df, "frameType", "trap", "Falle")
    df, color_dict = sort_and_get_colors(df, "frameType")
    color = list(color_dict.keys())
    if not df.empty:
        plot_cols[1].plotly_chart(
            vit.plotly_pie(
                df,
                "Anzahl",
                "frameType",
                title="Kartentyp",
                color=color,
                color_dict=color_dict,
            ),
            use_container_width=True,
        )
    # Extra Deck Ratio: feature claculation and plotting
    df = (
        st.session_state["extra_deck"]
        .groupby("frameType")["Anzahl"]
        .sum()
        .reset_index()
    )
    if not df.empty:
        df["frameType"] = df["frameType"].str.title()
        plot_cols[2].plotly_chart(
            vit.ploty_bar(df, "frameType", "Anzahl", True, "Extra-Deck"),
            use_container_width=True,
        )


def calculate_total_price(df: pd.DataFrame) -> float:
    """
    Method for calculationg the total price of a deck
    :param df: dataframe with prices and number of cards for price calculation
    """
    return np.round((df["Anzahl"] * df["price"]).sum(), 2)


def sort_and_get_colors(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Helper Method for plotting card types with right color
    :param df: dataframe with aggregated card type feature
    :param col: name of feature column
    :return dataframe in right order of feature for plotting
    """
    tmp = []
    color = {"Zauber": "green", "Falle": "red", "Monster": "orange"}
    dict_return = {}
    for string in color:
        idx = df[df[col] == string].index
        tmp.append(df.loc[idx])
        if not idx.empty:
            dict_return[string] = color[string]
    return pd.concat(tmp, ignore_index=True), dict_return


def reset_values(df: pd.DataFrame, col: str, old: float, new: float):
    """
    Method for reset all values in ca column with an new value
    :param df: datframe with data to reset
    :param col: column with data for reset
    :param old: value for reseting
    :param new: new value
    :return df: dataframe with resetted values
    """
    idx = df[df[col] == old].index
    df.loc[idx, col] = new
    return df
