import itertools

from DeckBuilderPackage.StatsPackage.calc_wsk_exact import calc_wsk_exact


def find_most_probable_combination(N, M, k, amount, hand_size=5):
    """
    Finds the most probable combination of events after k draws from M possible event groups.

    Parameters:
    N (int): Total number of particles.
    M (int): Number of possible event groups.
    k (int): Number of draws.
    amount (list): List of length M containing the probabilities of each event group.

    Returns:
    tuple: Tuple containing the most probable combination and its probability.
    """

    events_combinations = itertools.combinations_with_replacement(range(M), k)
    max_probability = 0.0
    most_probable_combination = ()
    combis = 0
    for combination in events_combinations:
        combis += 1
        prob_product = 1.0
        for group_idx in range(M):
            count = combination.count(group_idx)

            prob = calc_wsk_exact(N, amount[group_idx], hand_size, count)
            prob_product *= prob

        if prob_product > max_probability:
            max_probability = prob_product
            most_probable_combination = combination

    return most_probable_combination, max_probability, combis
