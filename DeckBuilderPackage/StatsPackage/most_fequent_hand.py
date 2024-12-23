import numpy as np
import pandas as pd
import random


def most_frequent_hand(M, groups, max_draws=1_000):
    """
    Method for calculation of the most frequent hand for the current deck
    using a max draw draws from the tagg distribution
    :param M: dataframe with number of cards for each tag
    :param groups: list of possible tags
    :return dataframe: results from draws
    """
    # init result data and deck array
    df = pd.DataFrame(index=range(max_draws), columns=groups)
    a = [(i + 1) * np.ones(num) for i, num in enumerate(M)]
    a = np.concatenate(a)
    # draw n times and count drawn tags
    for number in range(max_draws):
        random.shuffle(a)
        for idx in range(len(groups)):
            count = list(a[:5]).count(idx + 1)
            df.at[number, groups[idx]] = count
    return (
        df.mean()
        .reset_index()
        .rename(columns={0: "mittlere Anzahl", "index": "Kartentyp"})
    )
