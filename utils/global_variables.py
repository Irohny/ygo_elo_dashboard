import streamlit as st

DECK_ATTRIBUTES: list[str] = [
    "Attack",
    "Control",
    "Recovery",
    "Consistency",
    "Combo",
    "Resilience",
]
ELO_TIERS: list[str] = ["Tier 0", "Tier 1", "Tier 2", "Good", "Fun", "Kartenstapel"]
TOURNAMENT_TYPES: list[str] = ["Wanderpokal", "Local", "Fun Pokal", "Regional"]
TOURNAMENT_RESULTS: list[str] = ["Teilnahme", "Top", "Win"]
CARD_TAGS: list[str] = [
    "Rest",
    "Starter",
    "Brick",
    "Extender",
    "Handtrap",
    "Boardbreaker",
    "Engine",
    "Non-Engine",
]
DECK_PLAYSTYLS: list[str] = ["Attack", "Combo", "Midrange", "Tower", "Burn", "Control"]

# get color theme from streamlit layout
PRIMARY_COLOR = st.get_option("theme.primaryColor")
BACKGROUND_COLOR = st.get_option("theme.backgroundColor")
SECONDARY_BACKGROUND_COLOR = st.get_option("theme.secondaryBackgroundColor")
TEXT_COLOR = st.get_option("theme.textColor")
