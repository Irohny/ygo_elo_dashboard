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
    dbp.tagging_editor(cols[0], card_model, gv.CARD_TAGS)
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
tabs = st.tabs(["Eingabe", "Deck Breakdown", "Deck Analyse"])
create_input_tab(tabs[0])
dbp.create_breakdown(tabs[1])
dbp.create_analysis(tabs[2])
