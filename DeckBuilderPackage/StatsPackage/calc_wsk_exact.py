import scipy.special as scsp


def calc_wsk_exact(deck_size, amount_cards, hand_size, x):
    """
    Method for calculation the probabilty for min. x cards in a hand of handsize
    cards out of a deck with a given size using a binomial cofficient approach from
    combinatoric
    :param deck_size: number of cards in the deck
    :param card_amounts: amount of cards of each tag
    :param hand_size: number of drawn crads from deck
    :param x: min number of drawn cards of choosen type
    :return probaility: prop of min cards per typ
    """
    return (
        scsp.binom(amount_cards, x)
        * scsp.binom(deck_size - amount_cards, hand_size - x)
        / scsp.binom(deck_size, hand_size)
    )
