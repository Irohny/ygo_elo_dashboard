import sqlite3
import json
import hashlib


def map_tournament_json_to_db(json_file: str, database_name: str):
    """
    Überträgt eine JSON-Datei mit Turnierinformationen in eine SQLite-Datenbank.

    :param json_file: Pfad zur JSON-Datei, die die Turnierdaten enthält.
    :param database_name: Name der SQLite-Datenbankdatei.
    """
    # Datenbankschema
    table_name = "tournament"
    columns = {
        "ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "DeckID": "TEXT NOT NULL",
        "Datum": "INTEGER",
        "Win": "INTEGER",
        "Draw": "INTEGER",
        "Loss": "INTEGER",
        "Type": "TEXT",
        "Result": "TEXT",
    }

    # Verbindung zur Datenbank herstellen
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    connection_ygo = sqlite3.Connection(database_name)
    cursor_ygo = connection.cursor()

    # Tabelle erstellen
    columns_definition = ", ".join(
        [f"{col_name} {data_type}" for col_name, data_type in columns.items()]
    )
    create_table_query = (
        f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition});"
    )
    cursor.execute(create_table_query)

    # JSON-Datei laden
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Elemente auf Datenbankschema mappen und einfügen
    for element in data:
        deck_name = element.get("Deck", "")
        date = element.get("Date", "1970-01-01")

        # DeckID erstellen
        cursor_ygo.execute(
            f"SELECT DeckID FROM deck_statistics WHERE Deck = '{deck_name}'"
        )
        deck_id = cursor_ygo.fetchall()[0][0]

        # Datum in Integer-Format YYYYMMDD konvertieren
        formatted_date = int(date.replace("-", ""))

        # Daten mappen
        mapped_data = {
            "DeckID": deck_id,
            "Datum": formatted_date,
            "Win": int(element.get("Win", 0)),
            "Draw": int(element.get("Draw", 0)),
            "Loss": int(element.get("Loss", 0)),
            "Type": element.get("Mode", ""),
            "Result": element.get("Standing", ""),
        }

        # Einfüge-Query erstellen
        placeholders = ", ".join(["?" for _ in mapped_data])
        insert_query = f"INSERT INTO {table_name} ({', '.join(mapped_data.keys())}) VALUES ({placeholders});"

        # Daten einfügen
        cursor.execute(insert_query, tuple(mapped_data.values()))

    # Änderungen speichern und Verbindung schließen
    connection.commit()
    connection.close()


def generate_deck_id(deck_name: str, player_name: str) -> str:
    """
    Erzeugt eine eindeutige DeckID basierend auf dem Decknamen und dem Spielernamen.
    :param deck_name: Name des Decks.
    :param player_name: Name des Spielers.
    :return: Hash-ID als String.
    """
    unique_string = f"{deck_name}-{player_name}"
    return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()


def map_special_tournaments_to_db(json_file: str, database_name: str):
    """
    Mappt spezielle Turniereinträge aus einer JSON-Datei auf die Tabelle `tournament`.

    :param json_file: Pfad zur JSON-Datei, die die Liste der Elemente enthält.
    :param database_name: Name der SQLite-Datenbankdatei.
    """
    # Verbindung zur Datenbank herstellen
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    # Tabellenname und Spalten für die Tournament-Tabelle
    tournament_table_name = "tournament"
    tournament_columns = {
        "ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "DeckID": "TEXT NOT NULL",
        "Datum": "INTEGER",
        "Win": "INTEGER",
        "Draw": "INTEGER",
        "Loss": "INTEGER",
        "Type": "TEXT",
        "Result": "TEXT",
    }

    # Tabelle erstellen, falls sie nicht existiert
    tournament_columns_definition = ", ".join(
        [
            f"{col_name} {data_type}"
            for col_name, data_type in tournament_columns.items()
        ]
    )
    create_tournament_table_query = f"CREATE TABLE IF NOT EXISTS {tournament_table_name} ({tournament_columns_definition});"
    cursor.execute(create_tournament_table_query)

    # JSON-Dokument laden
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Standard-Datum
    default_date = 20190101

    for element in data:
        deck_name = element.get("Deck", "")
        player_name = element.get("Owner", "")
        deck_id = generate_deck_id(deck_name, player_name)

        # Fun Pokal, Wanderpokal prüfen und einfügen
        for special_field in ["Fun Pokal", "Wanderpokal"]:
            value = int(element.get(special_field, 0))
            print(deck_name, special_field, value)
            if value <= 0:
                continue
            for v in range(value):
                mapped_tournament_data = {
                    "DeckID": deck_id,
                    "Datum": default_date,
                    "Win": 0,
                    "Draw": 0,
                    "Loss": 0,
                    "Type": special_field,
                    "Result": "Win",
                }

                # Einfüge-Query für das Turnier erstellen
                tournament_placeholders = ", ".join(
                    ["?" for _ in tournament_columns if _ != "ID"]
                )
                insert_tournament_query = f"INSERT INTO {tournament_table_name} ({', '.join(k for k in tournament_columns if k != 'ID')}) VALUES ({tournament_placeholders});"

                # Daten einfügen
                cursor.execute(
                    insert_tournament_query, tuple(mapped_tournament_data.values())
                )

    # Änderungen speichern und Verbindung schließen
    connection.commit()
    connection.close()


# Beispielverwendung
if __name__ == "__main__":
    json_file_path = "DataModel/deta_backups/YuGiOh_Tournaments.json"  # Ersetze durch den Pfad zur JSON-Datei
    database_name = "DataModel/ygo_database.db"
    map_tournament_json_to_db(json_file_path, database_name)
    map_special_tournaments_to_db(
        "DataModel/deta_backups/YgoEloBase.json", database_name
    )
