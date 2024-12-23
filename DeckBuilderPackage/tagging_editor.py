import streamlit as st

def tagging_editor(st_obj: st, tags: list):
    """
    Method for tagging cards for further deck analysis
    :param st_obj: streamlit object for placing
    """
    st_obj = st_obj.container(border=True)
    tabs = st_obj.tabs(['Main Deck', 'Extra Deck', 'Side Deck'])
    
    head_cols = tabs[0].columns([4, 1])
    head_cols[0].markdown("Main Deck:")
    head_cols[1].markdown(
        f"""Monster: {count_frametypes(st.session_state['main_deck'], ['effect'])}/ 
                        Zauber: {count_frametypes(st.session_state['main_deck'], ['spell'])}/
                        Fallen: {count_frametypes(st.session_state['main_deck'], ['trap'])}"""
    )
    vis_data_editor(tabs[0], "main_deck", tags)

    head_cols = tabs[1].columns([4, 1])
    head_cols[0].markdown("Extra Deck")
    head_cols[1].markdown(
        f"""Synchro: {count_frametypes(st.session_state['extra_deck'], ['synchro'])}/ 
                        Fusion: {count_frametypes(st.session_state['extra_deck'], ['fusion'])}/
                        XYZ: {count_frametypes(st.session_state['extra_deck'], ['xyz'])}/
                        Link: {count_frametypes(st.session_state['extra_deck'], ['link'])}"""
    )
    vis_data_editor(tabs[1], "extra_deck", tags)

    head_cols = tabs[2].columns([4, 1])
    head_cols[0].markdown("Side Deck:")
    head_cols[1].markdown(
        f"""Monster: {count_frametypes(st.session_state['side_deck'], ['effect', 'synchro', 'xyz', 'fusion', 'link'])}/ 
                        Zauber: {count_frametypes(st.session_state['side_deck'], ['spell'])}/
                        Fallen: {count_frametypes(st.session_state['side_deck'], ['trap'])}"""
    )
    vis_data_editor(tabs[2], "side_deck", tags)

def vis_data_editor(st_obj: st, deck_part: str, tags:list[str]):
    configs = {
        "image":st.column_config.ImageColumn('Preview', width='small'),
        "Karte":st.column_config.TextColumn("Name", width='large'),
        "Anzahl":st.column_config.NumberColumn("Anzahl", width='medium', min_value=0, max_value=3, step=1),
        "Tag":st.column_config.SelectboxColumn("Tag", options=tags, width='medium'),
    }
    st_obj.data_editor(st.session_state[deck_part][configs.keys()], 
                    column_config=configs, 
                    hide_index=True,
                    on_change=changes2session_state,
                    key=f"data_editor_{deck_part}")

def changes2session_state():
    for deck_part in ['main_deck', 'side_deck', 'extra_deck']:
        if len(st.session_state[f'data_editor_{deck_part}']['edited_rows']) > 0:
            for idx, row in st.session_state[f'data_editor_{deck_part}']['edited_rows'].items():
                for col, val in row.items():
                    st.session_state[deck_part].at[idx, col] = val

def vis_and_tagging(
    st_obj: st,
    deck_part: str,
    tags: list,
    img_per_row: int = 6,
):
    """
    Method for visualization and modification of cards inside the deck
    """
    main_cols = st_obj.columns(img_per_row)
    for i, card in st.session_state[deck_part].iterrows():
        img = st.session_state['ygo_pro_model'].get_image(card["image"])
        pos, row = i % img_per_row, i // img_per_row
        cont = main_cols[pos].container(border=True)
        cont.text(card["Karte"])
        
        cont.image(img)
        modifier = cont.popover(f"{card['Anzahl']}, {card['Tag']}")
        modifier.number_input(
            "Anzahl",
            value=1,
            min_value=0,
            max_value=3,
            step=1,
            key=f"Anzahl {card['Karte']}",
            on_change=modify_deck_entry,
            args=(i, f"Anzahl {card['Karte']}", "Anzahl", deck_part),
        )
        modifier.selectbox(
            "Tag",
            tags,
            key=f"Tag {card['Karte']}",
            on_change=modify_deck_entry,
            args=(i, f"Tag {card['Karte']}", "Tag", deck_part),
        )


def modify_deck_entry(idx: int, key: str, column: str, deck_part: str):
    """
    Method for modifing the entries in a deck via the popover field
    :param idx: index in dataframe
    :param key: key of input widget
    :param column: column to modify in dataframe
    """
    # Drop from deck if anzahl = 0
    if column == "Anzahl" and st.session_state[key] == 0:
        st.session_state[deck_part].drop(index=idx, inplace=True)
        st.session_state[deck_part].reset_index(drop=True, inplace=True)
        return
    st.session_state[deck_part].at[idx, column] = st.session_state[key]


def count_frametypes(df, list_frame):
    """ """
    return df[df["frameType"].isin(list_frame)]["Anzahl"].sum()
