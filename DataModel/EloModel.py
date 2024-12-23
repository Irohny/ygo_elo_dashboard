import numpy as np
import pandas as pd
import datetime
import sqlite3
import streamlit as st
import hashlib

from DataModel.SQLModel import SQLModel


class EloModel(SQLModel):
    """
    Dataclass model for getting and processing data for visualization
    """

    def __init__(self, load_data: bool = True) -> None:
        self.db_name = "DataModel/ygo_database.db"
        self.table_name = "deck_statistics"
        self.df_default_meisterschaft = pd.DataFrame(
            0,
            index=["Wanderpokal", "Local", "Fun Pokal"],
            columns=["Teilnahme", "Top", "Win"],
        ).to_dict()

    @st.cache_data
    def get_all_decks(_self):
        """
        Method to get all data from the Database
        :use db: as detabase keys
        :return df_all: Dtaframe with all data like deck name, player and elo
        :return hist_cols: list of columns with historic elo data
        :return save_cols:list with all columns with further infos
        """
        sql = f"SELECT * FROM {_self.table_name}"
        return _self.exicute_sql_statement(sql)

    @st.cache_data
    def get_best_elo_by_player(_self, player: str):
        sql = f"""SELECT 
                    MAX(Elo) AS MaxElo,
                    MIN(Elo) AS MinElo
                FROM {_self.table_name}
                WHERE Player = '{player}' """
        return _self.exicute_sql_statement(sql)

    def update_elo_ratings(self, deck_id1, deck_id2, erg1, erg2, df_elo):
        """
        Method for updating Elo Rating and result stats after a new match
        """
        # get index
        idx1 = int(df_elo[df_elo["DeckID"] == deck_id1].index.to_numpy())
        idx2 = int(df_elo[df_elo["DeckID"] == deck_id2].index.to_numpy())
        # update siege/niederlagen
        if erg1 != erg2:
            df_elo.at[idx1, "Siege"] += erg1
            df_elo.at[idx1, "Niederlagen"] += erg2

            df_elo.at[idx2, "Siege"] += erg2
            df_elo.at[idx2, "Niederlagen"] += erg1
        else:
            df_elo.at[idx1, "Remis"] += erg1
            df_elo.at[idx2, "Remis"] += erg2
        # update matches
        df_elo.at[idx1, "Matches"] += 1
        df_elo.at[idx2, "Matches"] += 1
        # update dgp
        df_elo.at[idx1, "DGP"] = (
            df_elo.at[idx1, "DGP"] + 2 * df_elo.at[idx2, "Elo"]
        ) // 3
        df_elo.at[idx2, "DGP"] = (
            df_elo.at[idx2, "DGP"] + 2 * df_elo.at[idx1, "Elo"]
        ) // 3
        # update elo
        norm = erg1 + erg2
        score1 = erg1 / norm
        score2 = erg2 / norm

        # calculate winning probabilities
        # Formula from Chess ELO System
        alpha = 10 ** ((df_elo.at[idx1, "Elo"] - df_elo.at[idx2, "Elo"]) / 400)
        p1 = alpha / (alpha + 1)
        p2 = 1 - p1

        # Update ELO (Chess Formula)
        df_elo.at[idx1, "Elo"] += int(32 * (score1 - p1))
        df_elo.at[idx2, "Elo"] += int(32 * (score2 - p2))

        df_elo = self.__update_tiers(df_elo)

        for idx in [idx1, idx2]:
            sql = f"""UPDATE {self.table_name} 
					SET Elo = ?, 
						DGP = ?,
						Matches = ?,
						Siege = ?,
						Remis = ?,
						Niederlagen = ?,
						Tier = ?
					WHERE DeckID = ?
					"""
            values = (
                int(df_elo.at[idx, "Elo"]),
                int(df_elo.at[idx, "DGP"]),
                int(df_elo.at[idx, "Matches"]),
                int(df_elo.at[idx, "Siege"]),
                int(df_elo.at[idx, "Remis"]),
                int(df_elo.at[idx, "Niederlagen"]),
                str(df_elo.at[idx, "Tier"]),
                str(df_elo.at[idx, "DeckID"]),
            )
            self.exicute_sql_statement(sql, values)

    def __update_tiers(self, df):
        """
        Method for updating tier standing of a deck after a match
        """
        Label = ["Kartenstapel", "Fun", "Good", "Tier 2", "Tier 1", "Tier 0"]
        NCl = len(Label)

        mean = int(df["Elo"].mean())
        std = float(df["Elo"].std())
        bins = np.array(
            [0, mean - 2 * std, mean - std, mean, mean + std, mean + 2 * std, 10e6]
        )
        liste = []
        for jj in range(len(df)):
            for kk in range(NCl):
                if df.at[jj, "Elo"] >= bins[kk] and df.at[jj, "Elo"] <= bins[kk + 1]:
                    liste.append(Label[kk])
                    break

        df["Tier"] = liste
        return df

    def update_stats(self, deck_id: str, in_stats, modif_in, new_type):
        """
        Method for updating deck characteristica
        :param deck_choosen: name of the deck to update
        :param in_stats: info column to update
        :parm modif_in: value of the midification
        :param in_stats_type: update type of tthe choosen deck
        :param new_type: new_type to update
        :param df_elo: dataframe with all data
        :param save_cols: list with info columns
        :param hist_cols: columns with historic elo data
        :param db: key to database
        :return: nothing
        """
        values = []
        sql = f"UPDATE {self.table_name} SET "
        # update characteristics if choosen
        if in_stats:
            sql = sql + in_stats + " = ?"
            values.append(modif_in)
        # update deck type if choosen
        if new_type:
            if in_stats:
                sql = sql + ", "
            sql = sql + " Type = ?"
            values.append(new_type)
        sql = sql + f" WHERE DeckID = ?;"
        values.append(str(deck_id))
        self.exicute_sql_statement(sql, tuple(values))

    def insert_new_deck(self, new_deck: dict):
        """
        Function for inserting a new deck to the database
        :param deck: name of the deck
        :param name: name of the player
        :param atk: attack rating
        :param contr: control rating
        :param rec: recovery rating
        :param cons: consistency rating
        :param combo: combo rating
        :param resi: resilience rating
        :param typ; type of the deck
        :param db: key to database
        :return: nothing
        """
        new_deck["DeckID"] = self.generate_deck_id(new_deck["Deck"], new_deck["Player"])
        sql, values = self.insert_statement_from_dict(new_deck)
        print(sql)
        print(values)
        self.exicute_sql_statement(sql, values)

    def __clean_data(self, df):
        """
        Method for cleaning and calculation of new data with data from the database
        :param df: dataframe with data from the database
        :param hist_cols: list with all historic elo columns
        :return df: dataframe with data in right format and new calculated values
        """
        # columns with integer values
        cols_to_int = [
            "Elo",
            "dgp",
            "Wanderpokal",
            "Liga Pokal",
            "Fun Pokal",
            "Attack",
            "Combo",
            "Resilience",
            "Recovery",
            "Consistensy",
            "Control",
            "Matches",
            "Siege",
            "Remis",
            "Niederlage",
        ] + self.hist_cols
        # fill nulls and transform float data to interes
        for k in cols_to_int:
            df[k] = df[k].fillna(0).astype(float)
            df[k] = df[k].astype(int)
        # last 3 month
        df = self.__calcualte_past_stats(df, "Letzte 3 Monate", self.hist_cols, -1)
        # last 6 month
        df = self.__calcualte_past_stats(df, "Letzte 6 Monate", self.hist_cols, -2)
        # last year
        df = self.__calcualte_past_stats(df, "Letzte 12 Monate", self.hist_cols, -4)
        # calculate mean and std of the last year
        df["Mean 1 Year"] = np.nanmean(df[self.hist_cols[-5:-1] + ["Elo"]], 1)
        df["Mean 1 Year"] = df["Mean 1 Year"].astype(int)
        df["Std. 1 Year"] = np.nanstd(df[self.hist_cols[-5:-1] + ["Elo"]], 1)
        df["Std. 1 Year"] = df["Std. 1 Year"].astype(int)
        # calculate win rate
        df["Gewinnrate"] = 100 * np.round(
            df["Siege"] / (df["Siege"] + df["Remis"] + df["Niederlage"]), 2
        )

        df.fillna(0, inplace=True)
        # estimate if deck is active
        all_equal = df["Elo"] == df[self.hist_cols[-1]]
        for col in self.hist_cols[-4:-2]:
            all_equal = (all_equal) & (df["Elo"] == df[col])

        df["active"] = ~all_equal
        # generate Platz
        df.sort_values("Elo", ascending=False, inplace=True, ignore_index=True)
        df["EloPlatz"] = df.index.to_numpy() + 1
        return df

    def __calcualte_past_stats(self, df, new_column, hist_cols, past_idx):
        """
        Method for calcualting the history elo stats changes of the decks
        :param df: deck datatframe
        :param new_column: name of the new column
        :param hist_cols: history columns
        :param past_idx: index of the past date for change calculation
        :return df: deck dataframe with new featuer
        """
        df[new_column] = 0
        result = df[df[hist_cols[past_idx]] > 0].index
        df.loc[result, new_column] = (
            df.loc[result, "Elo"] - df.loc[result, hist_cols[past_idx]]
        )
        df[new_column] = df[new_column].astype(int)
        return df

    def __sort_hist_cols(self, hist_cols):
        """
        Method for sorting the list with historic elo in ascending order
        :param hist_cols: list with columns with historic elo
        :return out: sorted list with historic elo
        """
        t = pd.Series([int(f"{c[-4:]}{c[:2]}") for c in hist_cols]).sort_values(
            ascending=True, ignore_index=True
        )
        return [f"{date[-2:]}/{date[:4]}" for date in t.astype(str).values]

    def generate_deck_id(self, deck_name: str, player_name: str) -> str:
        """
        Erzeugt eine eindeutige DeckID basierend auf dem Decknamen und dem Spielernamen.
        :param deck_name: Name des Decks.
        :param player_name: Name des Spielers.
        :return: Hash-ID als String.
        """
        unique_string = f"{deck_name}-{player_name}"
        return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
