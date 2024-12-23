import streamlit as st
import pandas as pd

import VisualizationTools as vit
import DeckBuilderPackage.StatsPackage as stp


def create_analysis(tab: st):
    """
    Method for creating and handling the statistical analysis tab of the page
    :param tab: streamlit object for placing
    """
    # dont build tab if main deck is empty
    if st.session_state["main_deck"].empty:
        return
    # calculate stats from input
    deck_size = int(st.session_state["main_deck"]["Anzahl"].sum())
    amount_of_tags = st.session_state["main_deck"]["Tag"].nunique()  # len(Ne)
    counts_per_tag = (
        st.session_state["main_deck"].groupby("Tag")["Anzahl"].sum().astype(int)
    )  # Ne
    tag_list = counts_per_tag.index.to_list()
    image_list = (
        st.session_state["main_deck"].groupby("Tag")["image"].apply(get_first_image)
    )
    # analyse tab
    most_probable_comb, wsk, n_combis = stp.find_most_probable_combination(
        deck_size, amount_of_tags, 5, counts_per_tag
    )

    # metrics
    metrics = tab.columns(len(tag_list) + 1, vertical_alignment="bottom")
    metrics[0].container(border=True).metric("Deckgröße", deck_size)
    markers = []
    for idx, tag in enumerate(tag_list):
        if tag not in ["Rest"]:
            markers.append({"key": tag, "value": int(counts_per_tag[tag])})

        metrics[idx + 1].container(border=True).metric(
            tag,
            int(counts_per_tag[tag]),
            f"Anteil {100*counts_per_tag[tag]/deck_size:.1f}%",
        )
    # figures
    progbar = tab.progress(0, "Berechne Wahrscheinlichkeitentabelle")
    fig_cols = tab.columns([1, 2, 1])
    # header
    # fig_cols[0].markdown('### Wahrscheinlichkeiten von Kartentypen auf der Starthand in %')
    fig_cols[0].markdown("### Wahrscheinlichste Hand")
    fig_cols[0].text(f"Wahrscheinlichkeit: {wsk:.2f}%")
    fig_cols[0].text(f"{n_combis} Kombinationen")
    fig_cols[2].markdown("### Häufigste Hand")
    fig_cols[2].text("Durschnittliche Anzahl Karten pro Klasse nach 1000 Starthänden")
    # wsk table
    # going_first = stp.build_start_hand(deck_size, counts_per_tag, tag_list)
    # fig_cols[0].dataframe(going_first.round(2), hide_index=True)
    fig_cols[1].markdown("### Draw Wahrscheinlichkeiten")
    n = fig_cols[1].number_input(
        "Gewünschte Anzahl Karten auf der Hand",
        value=2,
        min_value=0,
        max_value=5,
        step=1,
    )
    props = stp.calc_probability_curves(deck_size, n)
    fig_cols[1].plotly_chart(vit.plot_probabilities(props, markers, n))
    # wsk hand
    progbar.progress(33, "Berechne wahrscheinlichste Hand")
    num_cards = []
    group = []
    for idx in range(amount_of_tags):
        num_cards.append(most_probable_comb.count(idx))
        group.append(tag_list[idx])
    fig = vit.image_piechart(num_cards, group, image_list)
    fig_cols[0].pyplot(fig, use_container_width=True)

    # frequent hand
    progbar.progress(66, "Berechne Häufigste Hand")
    df = stp.most_frequent_hand(counts_per_tag, tag_list)
    fig = vit.image_piechart(
        df["mittlere Anzahl"].to_list(), df["Kartentyp"].to_list(), image_list
    )
    fig_cols[2].pyplot(fig, use_container_width=True)
    progbar.empty()


def get_first_image(x: pd.Series) -> str:
    """
    Method for extracting the first given valid image
    link of a list of possible links
    :param x: list/Series of links
    :return item: valid link to image
    """
    for item in x:
        if not item:
            continue

        return item
