import sqlite3
import pandas as pd
import time


class SQLModel:
    def exicute_sql_statement(
        self, sql: str, values: tuple = None, exicution_flag: bool = False
    ):
        """tbd."""
        connection = sqlite3.connect(self.db_name)
        cursor = connection.cursor()
        df = pd.DataFrame()
        if sql.startswith("SELECT") or exicution_flag:
            df = pd.read_sql_query(sql, connection)

        elif (
            sql.startswith("INSERT")
            or sql.startswith("UPDATE")
            or sql.startswith("DELETE")
        ):
            cursor.execute(sql, values)
            connection.commit()

        connection.close()
        if not df.empty:
            return df

    def insert_statement_from_dict(self, input_dict: dict):
        placeholders = ", ".join(["?" for _ in range(len(input_dict))])
        sql = f"INSERT OR REPLACE INTO {self.table_name} ({', '.join(input_dict.keys())}) VALUES ({placeholders});"
        return sql, tuple(input_dict.values())
