import streamlit as st
import pandas as pd


class DeckModel:
    def __init__(self):
        # self.db = utils.connect2deta('YgoDeckBase', st.secrets['deck_key'])
        self.db_columns = ["key", "CardID", "CardTag", "Deck", "Location", "Player"]

    st.cache_data

    def load_deck(
        self,
        deck: str = "White Wood",
        player: str = "Chriss",
        ygo_db: pd.DataFrame = pd.DataFrame,
        convert_data: bool = True,
    ):
        """
        Deck Deck Data for a specific Player
        """
        # df = utils.fetch(self.db)
        if df.empty:
            return df
        df = df[(df["Deck"] == deck) & (df["Player"] == player)].reset_index(drop=True)
        if df.empty:
            return df
        if not convert_data:
            return df
        self.db2session_state(df, ygo_db)

    def update_deck(self, deck, player):
        """ """
        df_bevor = self.load_deck(deck, player, convert_data=False)
        df_new = self.__transform_session_state_to_db_format(deck, player)
        if df_new.empty:
            return
        # find cards for updating
        if not df_bevor.empty:
            df_new = df_new.merge(
                df_bevor[["key", "CardID"]],
                on="CardID",
                how="left",
                suffixes=["_1", None],
            )
            # find cards for inserting
            self.remove_from_db(
                df_bevor[~df_bevor["CardID"].isin(df_new["CardID"].unique())]
            )
        self.__update2db(df_new)
        st.toast("Deck Update Erfolgreich")

    def __update2db(self, df: pd.DataFrame):
        """ """
        df.reset_index(drop=True, inplace=True)
        for i, row in df.iterrows():
            row = row.to_dict()
            if row["key"] is None or pd.isna(row["key"]):
                del row["key"]
            utils.put(self.db, row)

    def remove_from_db(self, df: pd.DataFrame) -> None:
        """ """
        df.reset_index(drop=True, inplace=True)
        for i, row in df.iterrows():
            utils.delete(self.db, row["key"])

    def __transform_session_state_to_db_format(self, deck, player):
        """ """
        df_deck = []
        for part in ["main", "extra", "side"]:
            if st.session_state[f"{part}_deck"].empty:
                continue
            df_deck.append(
                self.dataframe2db(st.session_state[f"{part}_deck"], part, deck, player)
            )

        if len(df_deck) == 0:
            return pd.DataFrame()
        elif len(df_deck) == 1:
            return df_deck[0]
        df_deck = pd.concat(df_deck, ignore_index=True)
        return df_deck

    def db2session_state(self, df: pd.DataFrame, ygo_db: pd.DataFrame):
        """ """
        for part in ["main", "side", "extra"]:
            tmp = df[df["Location"] == part].reset_index(drop=True)
            if tmp.empty:
                continue
            tmp.rename(
                columns={"Amount": "Anzahl", "CardTag": "Tag", "CardID": "id"},
                inplace=True,
            )
            tmp = tmp.merge(
                ygo_db[["archetype", "name", "price", "image", "frameType", "id"]],
                on="id",
                how="left",
            )
            tmp.rename(columns={"name": "Karte"}, inplace=True)

            st.session_state[f"{part}_deck"] = tmp[
                [
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
            ]

    def dataframe2db(
        self, df: pd.DataFrame, part: str, deck: str, player: str
    ) -> pd.DataFrame:
        """ """
        tmp = df[["Anzahl", "id", "Tag", "key"]]
        tmp.rename(
            columns={"Anzahl": "Amount", "id": "CardID", "Tag": "CardTag"}, inplace=True
        )
        tmp["Player"] = player
        tmp["Location"] = part
        tmp["Deck"] = deck
        return tmp
