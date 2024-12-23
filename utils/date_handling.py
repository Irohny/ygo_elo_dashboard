from datetime import datetime


def convert_integer_to_date(date_int: int):
    """
    Konvertiert ein Datum im Format YYYYMMDD (als Integer) in ein datetime-Objekt.

    :param date_int: Datum als Integer (z.B., 20240705 für den 5. Juli 2024).
    :return: Ein datetime-Objekt.
    """
    date_str = str(date_int)
    return datetime.strptime(date_str, "%Y%m%d")


def get_current_date_as_integer():
    """
    Gibt das heutige Datum im Format YYYYMMDD als Integer zurück.
    :return: Heutiges Datum als Integer (YYYYMMDD)
    """
    today = datetime.now()
    return int(today.strftime("%Y%m%d"))
