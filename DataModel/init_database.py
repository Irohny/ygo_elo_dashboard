import sqlite3


def create_database_and_table(database_name: str, table_name: str, columns: dict):
    """
    Erstellt eine SQLite-Datenbank und eine Tabelle darin.

    :param database_name: Name der SQLite-Datenbankdatei (z. B. "example.db").
    :param table_name: Name der Tabelle, die erstellt werden soll.
    :param columns: Ein Dictionary mit Spaltennamen als Schlüssel und deren Datentypen als Werte
                    (z. B. {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "age": "INTEGER"}).
    """
    try:
        # Verbindung zur SQLite-Datenbank herstellen (erstellt die Datei, falls sie nicht existiert)
        connection = sqlite3.connect(database_name)
        cursor = connection.cursor()

        # SQL-Befehl zum Erstellen der Tabelle vorbereiten
        columns_definition = ", ".join(
            [f"{col_name} {data_type}" for col_name, data_type in columns.items()]
        )
        create_table_query = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition});"
        )

        # Tabelle erstellen
        cursor.execute(create_table_query)
        print(
            f"Tabelle '{table_name}' wurde erfolgreich in der Datenbank '{database_name}' erstellt."
        )

        # Änderungen speichern und Verbindung schließen
        connection.commit()
        connection.close()

    except sqlite3.Error as e:
        print(f"Fehler beim Erstellen der Tabelle {table_name}: {e}")


# Beispielverwendung
if __name__ == "__main__":
    database_name = "DataModel/ygo_database.db"
    # deck_statistics
    table_name = "deck_statistics"
    columns = {
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
    create_database_and_table(database_name, table_name, columns)
    # deck_history
    table_name = "deck_history"
    columns = {
        "ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "DeckID": "TEXT NOT NULL",
        "DGP": "INTEGER NOT NULL",
        "Elo": "INTEGER",
        "Spiele": "INTEGER",
        "Datum": "INTEGER",
    }
    create_database_and_table(database_name, table_name, columns)
    # tournament
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
    create_database_and_table(database_name, table_name, columns)
    # deck_list
    database_name = "DataModel/Sammlung.db"
    table_name = "deck_list"
    columns = {
        "ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "DeckID": "TEXT NOT NULL",
        "CollectionID": "TEXT NOT NULL",
        "Anzahl": "INTEGER",
        "CardID": "INTEGER",
    }
    create_database_and_table(database_name, table_name, columns)
    # collection
    table_name = "collection"
    columns = {
        "ID": "TEXT PRIMARY KEY",
        "Name": "TEXT",
        "Anzahl": "INTEGER",
        "Sets": "TEXT",
        "Preis": "INTEGER",
    }
    create_database_and_table(database_name, table_name, columns)
