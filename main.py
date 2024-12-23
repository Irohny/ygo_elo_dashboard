import streamlit as st

from DataModel.load_data import load_data

if __name__ == "__main__":
    st.set_page_config(
        page_title="YGO-Elo-Dashboard",
        page_icon=":trophy:",
        layout="wide",
    )
    if "deck_data" not in st.session_state:
        load_data()
    st.logo("Deck_Icons/ygo_icon.png", size="large")
    st.sidebar.markdown("Version: 2.0.0")
    pg = st.navigation(
        {
            "Elo Dshboard": [
                st.Page("views/StatsPage.py", title=" Hauptseite"),
                st.Page("views/PlayerPage.py", title=" Spieler"),
                st.Page("views/DeckComparisonPage.py", title=" Deckvergleich"),
                st.Page("views/DetailedInfoPage.py", title=" Deck Details"),
                st.Page(
                    "views/TablePage.py",
                    title=" Tabelle",
                ),
            ],
            "Deckbuilding": [
                st.Page(
                    "views/DeckBuilderPage.py",
                    title=" Deck Builder",
                )
            ],
            "Sammlung": [],
            "Einstellungen": [
                st.Page("views/InputPage.py", title=" Eingabe"),
                st.Page("views/LoginPage.py", title=" Login"),
            ],
        }
    )

    pg.run()
