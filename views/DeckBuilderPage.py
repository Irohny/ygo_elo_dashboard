import streamlit as st
import pandas as pd

from DataModel.YgoCradModel import YgoCardModel
import DeckBuilderPackage as dbp
from Authentificator.check_roles import is_player
import utils.global_variables as gv


def create_input_tab(st_obj: st):
    """
    Layout Method for the first tab (Input Tab)
    Structured in 3 columns
    1st Col: List input of deck parts
    2nd Col: Tagging of cards
    3rd Col: Seach for some cards
    :param st_obj: streamlit tab object for placing in the right tab
    """
    cols = st_obj.columns([3, 1])
    dbp.tagging_editor(cols[0], gv.CARD_TAGS)
    tabs = cols[1].tabs(["Suchfeld", "Nexus Import"])
    dbp.search_field_input(tabs[0], card_model)
    dbp.nexus_list_input(tabs[1], feat_cols, ygo_db)


##############################
# Build Page
##############################
feat_cols = [
    "key",
    "id",
    "image",
    "Karte",
    "Anzahl",
    "Tag",
    "price",
    "archetype",
    "frameType",
]
tags = [
    "Rest",
    "Starter",
    "Brick",
    "Extender",
    "Handtrap",
    "Boardbreaker",
    "Engine",
    "Non-Engine",
]

card_model = YgoCardModel()

# build page
if "main_deck" not in st.session_state:
    dbp.reset_session_state_for_decks(feat_cols)

ygo_db = card_model.get_data()


header_col = st.columns([1.5, 1])
header_col[0].header(":trophy: Deck Builder :trophy:")
dbp.database_handler(
    header_col[1],
    st.session_state["deck_data"].copy(),
    ygo_db,
    feat_cols,
)
segment = st.segmented_control('BlaBla', ["Eingabe", "Deck Breakdown", "Deck Analyse"], 
                                default='Eingabe', 
                                label_visibility='collapsed')
if segment == 'Eingabe':
    create_input_tab(st)
elif segment == 'Deck Breakdown':
    dbp.create_breakdown(st)
elif segment == 'Deck Analyse':
    dbp.create_analysis(st)
else:
    st.error('Keine Auswahl getroffen.')
