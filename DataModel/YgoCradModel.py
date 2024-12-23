import requests
from PIL import Image
from io import BytesIO
import numpy as np
import streamlit as st
import pandas as pd


class YgoCardModel:
    """
    Data Model for communication and feature extraction from yugioh pro api
    Endpoints:
    1. get_data: fetch all data from ygo pro and preprocess them
    2. fuzzy_seach: search for spezific cards
    3. get_image: get image of a card from ygo pro
    """

    def __init__(self):
        self.base_url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
        self.keep_cols = [
            "archetype",
            "name",
            "price",
            "image",
            "frameType",
            "id",
            "key",
        ]

    @st.cache_data
    def get_data(_self):
        """
        Api Call of the yugioh pro deck database of all cards
        :return ygo_db: DataFrame with feature of interes
        -----------------
        Calculated Feature:
        1. min price of cards
        2. image path of card
        """
        respons = requests.get(_self.base_url)
        ygo_db = pd.json_normalize(respons.json()["data"])
        ygo_db["price"] = ygo_db["card_prices"].apply(_self.__get_price)
        ygo_db["image"] = ygo_db["card_images"].apply(_self.__get_image_link)
        ygo_db["key"] = None
        return ygo_db[_self.keep_cols]

    @st.cache_data
    def fuzzy_search(_self, string: str) -> pd.DataFrame:
        # sourcery skip: simplify-empty-collection-comparison, simplify-str-len-comparison
        """
        Method for searching cards in pro deck datbase based on the fuzzy search endpoint
        :param string: string for searching
        :return cards: dataframe with matching cards of search
        """
        # return empty dataframe if string is empty
        if not string:
            return pd.DataFrame(columns=["image", "Hinzufügen", "name"])
        # get results of search
        url = f"{_self.base_url}?fname={string}"
        respons = requests.get(url)
        cards = pd.json_normalize(respons.json()["data"])
        if cards.empty:
            return cards
        # calculate feature of search results
        cards["price"] = cards["card_prices"].apply(_self.__get_price)
        cards["image"] = cards["card_images"].apply(_self.__get_image_link)
        cards["Hinzufügen"] = False
        cards["Anzahl"] = 1
        cards["Tag"] = "Rest"
        cards["Karte"] = cards["name"]
        cards["key"] = None
        if "archetype" not in cards.columns:
            cards["archetype"] = None
        return cards[
            [
                "Karte",
                "Tag",
                "name",
                "price",
                "archetype",
                "image",
                "Anzahl",
                "Hinzufügen",
                "frameType",
                "id",
                "key",
            ]
        ]

    def get_image(self, url):
        """
        Method for getting the image for displaying from a link
        :param url: link to the image
        :return array: array with the image data
        """
        res = requests.get(url)
        if res.status_code == 200:
            img_arr = Image.open(BytesIO(res.content))
            img_arr.thumbnail((256, 256))
        return np.array(img_arr)

    def __get_image_link(self, x):
        """
        Method for extraction of the image link of the ygo pro deck
        api elment
        """
        return str(x[0]["image_url_cropped"])

    def __get_price(self, x):
        """
        Method for getting the minial price of the ygo pro deck
        api element
        """
        y = []
        for idx in x:
            if "cardmarket_price" in idx:
                y.append((float(idx["cardmarket_price"])))
            else:
                y.append(0)
        return min(y)
