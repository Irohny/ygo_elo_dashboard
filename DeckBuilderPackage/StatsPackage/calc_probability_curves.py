import numpy as np
import pandas as pd

from DeckBuilderPackage.StatsPackage.calc_wsk_exact import calc_wsk_exact


def calc_probability_curves(deck_size, number_wanted):
    """
    deck_size = size of deck
    number_wanred = amount of cards wanted too draw in start hand of size 5
    """
    df = pd.DataFrame(columns=["Cards", "Wsk_exact", "Wsk_min"], index=range(deck_size))
    for i in range(deck_size):
        df.at[i, "Cards"] = i

        # Berechnung der Wahrscheinlichkeit für genau `number_wanted` Karten
        df.at[i, "Wsk_exact"] = (
            calc_wsk_exact(deck_size, i, 5, number_wanted) if i >= number_wanted else 0
        )

        # Berechnung der Wahrscheinlichkeit für mindestens `number_wanted` Karten
        tmp = 0
        for k in range(number_wanted, 6):  # von `number_wanted` bis `5` (inklusive)
            if i < k:
                continue
            tmp += calc_wsk_exact(deck_size, i, 5, k)
        df.at[i, "Wsk_min"] = tmp
    return df
