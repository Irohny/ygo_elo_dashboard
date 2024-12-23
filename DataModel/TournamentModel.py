import streamlit as st
import pandas as pd
import numpy as np

from DataModel.SQLModel import SQLModel
import utils.global_variables as gv


class TournamentModel(SQLModel):
    def __init__(self, load_data=True):
        self.db_name = "DataModel/ygo_database.db"
        self.table_name = "tournament"
        if load_data:
            # get tournmaent data
            self.df = self.exicute_sql_statement(f"SELECT * FROM {self.table_name}")
            if self.df is not None:
                self.df["Date"] = self.df["Datum"].astype(int)
                self.__group_results()
                self.__calculate_tournament_score()

    def get_tournament_scores(self):
        return self.df_score

    def get_all_tournaments(self):
        return self.df

    def get_tournament_result_of_deck_by_id(self, deck_id: str):
        results = {
            "local_wins": 0,
            "local_tops": 0,
            "wanderpokal": 0,
            "fun_pokal": 0,
            "Win": 0,
            "Loss": 0,
            "Draw": 0,
        }
        if self.df.empty:
            return results
        tournament = self.df[self.df["DeckID"] == deck_id]
        if tournament.empty:
            return results
        results["local_wins"] = sum(
            (tournament["Type"] == "Local") & (tournament["Result"] == "Win")
        )
        results["local_tops"] = sum(
            (tournament["Type"] == "Local") & (tournament["Result"] == "Top")
        )
        results["wanderpokal"] = sum(
            (tournament["Type"] == "Wanderpokal") & (tournament["Result"] == "Win")
        )
        results["fun_pokal"] = sum(
            (tournament["Type"] == "Fun Pokal") & (tournament["Result"] == "Win")
        )
        results["Win"] = tournament["Win"].sum()
        results["Draw"] = tournament["Draw"].sum()
        results["Loss"] = tournament["Loss"].sum()
        return results

    def __group_results(
        self,
    ) -> None:
        """
        Method for combining torunament dat to deck data for local results
        """
        df_agg = []
        for tour in gv.TOURNAMENT_TYPES:
            for res in gv.TOURNAMENT_RESULTS:
                filtered = self.__group_results_for_tournament(res, tour)
                if filtered.empty:
                    continue
                df_agg.append(filtered)
        self.df_agg = pd.concat(df_agg, ignore_index=True)

    def __group_results_for_tournament(self, result: pd.DataFrame, mode: str):
        """
        Method for merging tournament data to deck data
        :param result: tournament satnding
        :param new_name: name of the tournament standing column
        :return result:merged dataframe
        """
        group_cols = ["Win", "Draw", "Loss", "DeckID"]
        # get filter for input
        filtered = self.df[(self.df["Result"] == result) & (self.df["Type"] == mode)]
        if filtered.empty:
            return pd.DataFrame()
        filtered = (
            filtered[group_cols]
            .groupby("DeckID")
            .agg({"Win": sum, "Draw": sum, "Loss": sum, "DeckID": "count"})
        )
        filtered.rename(columns={"DeckID": "Anzahl"}, inplace=True)
        filtered["Result"] = result
        filtered["Type"] = mode
        filtered.reset_index(inplace=True)
        return filtered

    def insert_tournament(self, result_dict):
        """
        Method for updating the tournament column of a spezific deck
        :param result_dict: diconary with tournament results
        """

        result_dict["Date"] = int(result_dict["Date"].replace("-", ""))
        mapped_data = {
            "DeckID": result_dict["DeckID"],
            "Datum": result_dict["Date"],
            "Win": int(result_dict.get("Win", 0)),
            "Draw": int(result_dict.get("Draw", 0)),
            "Loss": int(result_dict.get("Loss", 0)),
            "Type": result_dict.get("Mode", ""),
            "Result": result_dict.get("Standing", ""),
        }

        sql, values = self.insert_statement_from_dict(mapped_data)
        self.exicute_sql_statement(sql, values)

    def __calculate_tournament_score(
        self,
    ):
        # tournament standings
        self.df_score = self.df[
            ~self.df["Type"].isin(["Wanderpokal", "Fun Pokal"])
        ].reset_index(drop=True)
        self.df_score = self.df_score[
            ["DeckID", "Win", "Loss", "Draw", "Result"]
        ].copy()
        self.df_score["Turniere"] = 0
        self.df_score["Top-Rate"] = 0
        self.df_score["Match-Win-Rate"] = 0
        self.df_score["Result"] = self.df_score["Result"].apply(
            self.__tournament_points
        )
        self.df_score["Points"] = (
            3 * self.df_score["Win"] + self.df_score["Draw"] + self.df_score["Result"]
        )
        self.df_score = self.df_score.groupby(by="DeckID").sum()

        tops = self.df[self.df["Result"] == "Top"].groupby(by="DeckID").count()
        counts = self.df[["Result", "DeckID"]].groupby(by="DeckID").count()

        for idx in self.df_score.index:
            n = counts.at[idx, "Result"]
            if n == 0:
                continue
            if idx in tops.index:
                self.df_score.at[idx, "Top-Rate"] = np.round(
                    100 * tops.at[idx, "Result"] / n, 2
                )
            self.df_score.at[idx, "Turniere"] = n

        self.df_score["Points"] /= self.df_score["Turniere"]
        self.df_score["Points"] = np.round(self.df_score["Points"], 2)

        total_tourn_games = (
            self.df_score["Win"] + self.df_score["Draw"] + self.df_score["Loss"]
        )
        self.df_score["Match-Win-Rate"] = self.df_score["Win"] / total_tourn_games
        self.df_score["Match-Win-Rate"] = np.round(self.df_score["Match-Win-Rate"], 2)

        self.df_score.sort_values(by="Points", ascending=False, inplace=True)
        self.df_score.reset_index(inplace=True)
        self.df_score["Platz"] = self.df_score.index.to_numpy() + 1
        self.df_score.rename(
            columns={"Win": "Tourn_Win", "Loss": "Tourn_Loss", "Draw": "Tourn_Draw"},
            inplace=True,
        )

    def __tournament_points(self, x):
        """ """
        coding = {"Win": 10, "Top": 5, "Teilnahme": 1}
        return coding[x]
