import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt

import utils.global_variables as gv


def deck_density(values: list[float]):
    """ """
    H, bins = do_histogram(values, b=10, d=False)

    mean = np.mean(values)
    std = np.std(values)

    x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
    transparency = 0.3

    fig = plt.figure(figsize=(6, 6))
    fig.set_figheight(10)
    fig.set_figwidth(15)

    # create grid for different subplots
    spec = gridspec.GridSpec(ncols=1, nrows=2, hspace=0, height_ratios=[6, 1])
    axs = fig.add_subplot(spec[0])
    axs1 = fig.add_subplot(spec[1])

    axs2 = axs.twinx()
    # Verteilungsfunktion
    axs2.plot(x, gauss(x, mean, std), label="PDF", color="gray")

    range_data = [
        {"Label": "Sign. Bereich", "Color": "r", "Upper": -2, "Lower": -4},
        {"Label": "95% Interv.", "Color": "g", "Upper": -1, "Lower": -2},
        {"Label": "68% Interv.", "Color": "b", "Upper": 0, "Lower": -1},
        {"Label": "68% Interv.", "Color": "b", "Upper": 1, "Lower": 0},
        {"Label": "95% Interv.", "Color": "g", "Upper": 2, "Lower": 1},
        {"Label": "Sign. Bereich", "Color": "r", "Upper": 4, "Lower": 2},
    ]

    for interv in range_data:
        lower = mean + interv["Lower"] * std
        upper = mean + interv["Upper"] * std
        # set colored inrervalls
        axs2.fill_between(
            np.linspace(lower, upper, 100),
            gauss(np.linspace(lower, upper, 100), mean, std),
            color=interv["Color"],
            label=interv["Label"] if interv["Upper"] > 0 else None,
            alpha=transparency,
        )

        axs1.fill_between(
            [lower, upper], [1, 1], color=interv["Color"], alpha=transparency
        )

        # set gray boundary intervalls
        if interv["Lower"] < -2:
            continue

        axs1.plot([lower, lower], [0, 1], color="gray")
        axs2.plot(
            [lower, lower],
            [0, gauss(lower, mean, std)],
            color="gray",
        )

        text_offset = 0 if interv["Lower"] >= 0 else 20
        axs2.text(
            lower - text_offset,
            gauss(lower, mean, std),
            "%.f" % (lower),
            fontsize=15,
            color=gv.TEXT_COLOR,
        )

    # Data
    axs.bar(
        bins,
        H,
        abs(bins[1] - bins[0]),
        alpha=0.65,
        color="gray",
        label="Deck Histogram",
    )

    # Layout
    axs.grid()
    axs.legend(loc="upper left", fontsize=15)
    axs2.legend(loc="upper right", fontsize=15)
    axs2.set_ylim([0, 1.1 * np.max(gauss(x, mean, std))])
    axs.set_xlim([mean - 4 * std, mean + 4 * std])

    offset1 = 25
    offset = 20

    axs1.xaxis.label.set_color(gv.TEXT_COLOR)
    axs1.yaxis.label.set_color(gv.TEXT_COLOR)
    axs2.xaxis.label.set_color(gv.TEXT_COLOR)
    axs2.yaxis.label.set_color(gv.TEXT_COLOR)

    for orientation in ["bottom", "left", "right", "top"]:
        axs1.spines[orientation].set_color(gv.TEXT_COLOR)
        axs2.spines[orientation].set_color(gv.TEXT_COLOR)
    axs1.tick_params(colors=gv.TEXT_COLOR, which="both")
    axs2.tick_params(colors=gv.TEXT_COLOR, which="both")

    tier_bounds_text = [3, 1.5, 0.5, -0.5, -1.5, -3]  # in sigmas
    tier_bounds = [4, 2, 1, 0, -1, -2, -4]
    for i, category_label in enumerate(gv.ELO_TIERS):
        axs1.text(
            mean + tier_bounds_text[i] * std - offset,
            0.65,
            category_label,
            fontsize=20,
            color=gv.TEXT_COLOR,
        )

        N = len(
            values[
                (values <= mean + tier_bounds[i] * std)
                & (values > mean + tier_bounds[i + 1] * std)
            ]
        )
        axs1.text(
            mean + tier_bounds_text[i] * std - offset1,
            0.25,
            "%.f Decks" % (N),
            fontsize=20,
            color=gv.TEXT_COLOR,
        )

    axs1.set_xlim([mean - 4 * std, mean + 4 * std])
    axs1.set_xlabel("ELO", fontsize=20, color=gv.TEXT_COLOR)
    axs1.set_ylim([0, 1])
    axs1.set(yticklabels=[])  # remove the tick labels
    axs1.tick_params(left=False)  # remove the ticks
    axs2.set(yticklabels=[])  # remove the tick labels
    axs2.tick_params(left=False)  # remove the ticks
    axs1.grid()
    axs1.tick_params(axis="x", labelsize=15, color=gv.TEXT_COLOR)
    axs1.tick_params(axis="y", labelsize=15, color=gv.TEXT_COLOR)
    axs2.tick_params(axis="y", labelsize=15, color=gv.TEXT_COLOR)
    axs.tick_params(axis="y", labelsize=15, color=gv.TEXT_COLOR)
    return fig


def gauss(x, m, s):
    """ """
    return 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-((x - m) ** 2) / (4 * s**2))


def do_histogram(data, b=10, d=True):
    """ """
    counts, bins = np.histogram(data, b, density=d)

    n = np.size(bins)
    cbins = np.zeros(n - 1)

    for ii in range(n - 1):
        cbins[ii] = (bins[ii + 1] + bins[ii]) / 2

    return counts, cbins
