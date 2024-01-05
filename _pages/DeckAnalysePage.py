import streamlit as st
import requests
import pandas as pd


class DeckAnalysePage():
    """
    Class of the Streamlit YuGiOH! Dashboard that analyses a YGOPRO Deck via import a link
    """
    def __init__(self,):
        """
        Initialisation Method
        """
        self.__build_page()

    def __build_page(self, ):
        link = st.text_input('Insert YGO Pro Link')
        boolean = False
        if boolean:
            st.success()
        else:
            st.error('False Link')