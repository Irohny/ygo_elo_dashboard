import sqlite3
import json
import hashlib


def generate_deck_id(deck_name: str, player_name: str) -> str:
    """
    Erzeugt eine eindeutige DeckID basierend auf dem Decknamen und dem Spielernamen.
    :param deck_name: Name des Decks.
    :param player_name: Name des Spielers.
    :return: Hash-ID als String.
    """
    unique_string = f"{deck_name}-{player_name}"
    return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()


def map_and_store_json_to_db(json_file: str, database_name: str):
    """
    Lädt ein JSON-Dokument, mappt die Attribute auf das gegebene Datenbankschema und speichert die Daten in einer SQLite-Datenbank.

    :param json_file: Pfad zur JSON-Datei, die die Liste der Elemente enthält.
    :param database_name: Name der SQLite-Datenbankdatei.
    """
    # Datenbankschema
    statistics_table_name = "deck_statistics"
    statistics_columns = {
        "DeckID": "TEXT PRIMARY KEY",
        "Deck": "TEXT NOT NULL",
        "Player": "TEXT NOT NULL",
        "Elo": "INTEGER",
        "DGP": "INTEGER",
        "Tier": "TEXT NOT NULL",
        "Matches": "INTEGER",
        "Siege": "INTEGER",
        "Remis": "INTEGER",
        "Niederlagen": "INTEGER",
        "Type": "TEXT",
        "Attack": "INTEGER",
        "Combo": "INTEGER",
        "Consistency": "INTEGER",
        "Control": "INTEGER",
        "Recovery": "INTEGER",
        "Resilience": "INTEGER",
    }

    history_table_name = "deck_history"
    history_columns = {
        "DeckID": "TEXT NOT NULL",
        "Elo": "INTEGER",
        "DGP": "INTGER",
        "Spiele": "INTEGER",
        "Datum": "INTEGER",
    }

    # Verbindung zur Datenbank herstellen
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    # Tabellen erstellen
    statistics_columns_definition = ", ".join(
        [
            f"{col_name} {data_type}"
            for col_name, data_type in statistics_columns.items()
        ]
    )
    create_statistics_table_query = f"CREATE TABLE IF NOT EXISTS {statistics_table_name} ({statistics_columns_definition});"
    cursor.execute(create_statistics_table_query)

    history_columns_definition = ", ".join(
        [f"{col_name} {data_type}" for col_name, data_type in history_columns.items()]
    )
    create_history_table_query = f"CREATE TABLE IF NOT EXISTS {history_table_name} ({history_columns_definition});"
    cursor.execute(create_history_table_query)

    # JSON-Dokument laden
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Elemente auf Datenbankschema mappen und einfügen
    for element in data:
        deck_name = element.get("Deck", "")
        player_name = element.get("Owner", "")
        deck_id = generate_deck_id(deck_name, player_name)

        # Deck-Statistiken mappen
        mapped_statistics_data = {
            "DeckID": deck_id,
            "Deck": deck_name,
            "Player": player_name,
            "Elo": int(element.get("Elo", 0)),
            "DGP": int(element.get("dgp", 0)),
            "Tier": element.get("Tier", ""),
            "Matches": int(element.get("Matches", 0)),
            "Siege": int(element.get("Siege", 0)),
            "Remis": int(element.get("Remis", 0)),
            "Niederlagen": int(element.get("Niederlage", 0)),
            "Type": element.get("Type", ""),
            "Attack": int(element.get("Attack", 0)),
            "Combo": int(element.get("Combo", 0)),
            "Consistency": int(element.get("Consistensy", 0)),
            "Control": int(element.get("Control", 0)),
            "Recovery": int(element.get("Recovery", 0)),
            "Resilience": int(element.get("Resilience", 0)),
        }

        # Einfüge-Query für Statistiken erstellen
        statistics_placeholders = ", ".join(["?" for _ in statistics_columns])
        insert_statistics_query = f"INSERT OR REPLACE INTO {statistics_table_name} ({', '.join(statistics_columns.keys())}) VALUES ({statistics_placeholders});"

        # Statistiken einfügen
        cursor.execute(insert_statistics_query, tuple(mapped_statistics_data.values()))

        # Deck-Historie mappen und einfügen
        if "History" in element:
            for date, elo in element["History"].items():
                formatted_date = date.split("/")
                formatted_date = formatted_date[1] + formatted_date[0] + "01"
                # Datum in das Format YYYYMM01 konvertieren
                if int(elo) <= 0:
                    continue

                mapped_history_data = {
                    "DeckID": deck_id,
                    "Elo": int(elo),
                    "DGP": int(element.get("dgp", 0)),
                    "Spiele": int(element.get("Matches", 0)),
                    "Datum": formatted_date,
                }

                # Einfüge-Query für Historie erstellen
                history_placeholders = ", ".join(["?" for _ in history_columns])
                insert_history_query = f"INSERT INTO {history_table_name} ({', '.join(history_columns.keys())}) VALUES ({history_placeholders});"

                # Historie einfügen
                cursor.execute(
                    insert_history_query, tuple(mapped_history_data.values())
                )

    # Änderungen speichern und Verbindung schließen
    connection.commit()
    connection.close()


# Beispielverwendung
if __name__ == "__main__":
    json_file_path = "DataModel/deta_backups/YgoEloBase.json"
    database_name = "DataModel/ygo_database.db"
    map_and_store_json_to_db(json_file_path, database_name)
