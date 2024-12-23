import streamlit as st
import datetime
import pandas as pd
from .SQLModel import SQLModel


class HistoryModel(SQLModel):
    def __init__(
        self,
    ):
        self.db_name = "DataModel/ygo_database.db"
        self.table_name = "deck_history"

    def get_all_decks(self):
        return self.exicute_sql_statement(f"SELECT * FROM {self.table_name}")

    def get_history_of_deck(self, deck_id_list: str):
        """Get history of a deck."""
        sql = f"SELECT * FROM {self.table_name} WHERE DeckID = {deck_id}"
        df = self.exicute_sql_statement(sql)
        df["Datum"] = (
            df["Datum"]
            .astype(int)
            .sort_values("Datum", ascending=True, ignore_index=True)
        )
        return df

    def get_history_statistics(
        self,
    ):
        """Get  statistics of hiostory table."""
        sql = f"""SELECT * 
                FROM {self.table_name} 
                WHERE Elo = (SELECT MAX(Elo) FROM {self.table_name}) 
                OR Elo = (SELECT MIN(Elo) FROM {self.table_name});"""
        return self.exicute_sql_statement(sql)

    def get_paused_deckIDs(self):
        sql = f"""WITH RecentHistory AS (
                    SELECT DeckID, Elo, COUNT(DISTINCT Elo) AS EloChanges
                    FROM {self.table_name}
                    WHERE Datum BETWEEN strftime('%Y%m', 'now', '-12 months') || '01'
                                    AND strftime('%Y%m', 'now') || '01'
                    GROUP BY DeckID
                )
                SELECT DeckID
                FROM RecentHistory
                WHERE EloChanges = 1;"""
        return self.exicute_sql_statement(sql, exicution_flag=True)

    @st.cache_data
    def get_best_elo_by_player(_self, player: str):
        sql = f"""SELECT 
                    MAX(Elo) AS MaxElo,
                    MIN(Elo) AS MinElo
                FROM {_self.table_name}
                WHERE Elo > 0
                AND DeckID IN (
                    SELECT DeckID 
                    FROM deck_statistics
                    WHERE Player = '{player}'
                )
                """
        return _self.exicute_sql_statement(sql)

    def get_elo_history_of_deck_by_id(self, deck_id: str):
        sql = f"SELECT Elo, Datum FROM {self.table_name} WHERE DeckID = '{deck_id}'"
        df = self.exicute_sql_statement(sql)
        if (df is not None) and (not df.empty):
            return df
        return pd.DataFrame(columns=["Elo", "Datum"])

    def elo_changes(self):
        sql = """WITH comparison_dates AS (
                    SELECT 
                        DATE('now') AS today,
                        DATE('now', '-3 months') AS three_months_ago,
                        DATE('now', '-6 months') AS six_months_ago,
                        DATE('now', '-12 months') AS twelve_months_ago
                ),
                history AS (
                    SELECT 
                        ds.DeckID,
                        ds.Elo AS current_elo,
                        MAX(CASE WHEN dh.Datum <= strftime('%Y%m%d', three_months_ago) THEN dh.Elo END) AS elo_3_months,
                        MAX(CASE WHEN dh.Datum <= strftime('%Y%m%d', six_months_ago) THEN dh.Elo END) AS elo_6_months,
                        MAX(CASE WHEN dh.Datum <= strftime('%Y%m%d', twelve_months_ago) THEN dh.Elo END) AS elo_12_months
                    FROM 
                        deck_statistics ds
                    JOIN 
                        deck_history dh ON ds.DeckID = dh.DeckID
                    CROSS JOIN 
                        comparison_dates
                    GROUP BY 
                        ds.DeckID
                )
                SELECT 
                    DeckID,
                    COALESCE(current_elo - elo_3_months, 0) AS elo_change_3_months,
                    COALESCE(current_elo - elo_6_months, 0) AS elo_change_6_months,
                    COALESCE(current_elo - elo_12_months, 0) AS elo_change_12_months
                FROM 
                    history;
        """
        return self.exicute_sql_statement(sql, exicution_flag=True)

    def update_history(self, df):
        """
        Method for increase the historic elo columns with the actual elo
        :param df: dataframe with all data
        :param hist_cols: list with historic elo columns
        :param save_cols: list with deck info columns
        :param db: key to database
        :return: nothing
        """
        # get actual date and modify layout
        date = datetime.date.today()
        date = date.strftime("%m/%Y").split("/")
        date = date[1] + date[0] + "01"

        ids_to_update = self.get_updated_deck_ids()
        for deck_id in ids_to_update["DeckID"].to_list():
            deck = df[df["DeckID"] == deck_id].iloc[0]
            input_dict = {
                "DeckID": deck_id,
                "Elo": int(deck["Elo"]),
                "Spiele": int(deck["Matches"]),
                "DGP": int(deck["DGP"]),
                "Datum": int(date),
            }
            sql, values = self.insert_statement_from_dict(input_dict)
            self.exicute_sql_statement(sql, values)

    def get_updated_deck_ids(self):
        sql = """ 
            WITH LatestHistory AS (
                SELECT DeckID, Elo, Spiele, MAX(Datum) AS Datum
                FROM deck_history
                GROUP BY DeckID
            )
            SELECT ds.DeckID
            FROM deck_statistics ds
            JOIN LatestHistory lh
                ON ds.DeckID = lh.DeckID
            WHERE ds.Elo != lh.Elo OR ds.Matches != lh.Spiele;
            """
        return self.exicute_sql_statement(sql, exicution_flag=True)
