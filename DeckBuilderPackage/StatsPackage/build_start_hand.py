import numpy as np
import pandas as pd

from DeckBuilderPackage.StatsPackage.calc_wsk_exact import calc_wsk_exact


def build_start_hand(deck_size, card_amounts, groups) -> pd.DataFrame:
    """
    Method for calculating the probabilty of the min amount of cards on the
    starting hand for the cards of all choosen tags
    :param deck_size: numberof cards in the deck
    :param card_amounts: amount of cards of each tag
    :param groups: choosen tags
    """
    # define result variables for saving
    df_first = pd.DataFrame(columns=groups, index=range(6))
    df_first["Anzahl Karten"] = [
        "kein",
        "min. 1",
        "min. 2",
        "min. 3",
        "min. 4",
        "min. 5",
    ]
    # calculate probabilties using combinatoric approach
    for cat, amount in zip(groups, card_amounts):
        df_first[cat] = calc_min_hand(deck_size, amount, 5)
    df_first[groups] = df_first[groups] * 100
    df_first = df_first.reindex(sorted(df_first.columns), axis=1)
    return df_first


def calc_min_hand(deck_size, amount_cards, draws):
    """
    Mehtod for calcualting the probabilty by the given card tag distribution
    for n drwn cards
    :param deck_size: numberof cards in the deck
    :param card_amounts: amount of cards of each tag
    :param drws: number of drawn cards 5 for strating hand
    """
    wsk = np.zeros(draws + 1)
    tmp = np.zeros(draws + 1)
    for i in range(draws + 1):
        tmp[i] = calc_wsk_exact(deck_size, amount_cards, draws, i)
        if i == 0:
            wsk[0] = tmp[0]
        else:
            wsk[i] = 1 - np.sum(tmp[:i])
    return wsk
