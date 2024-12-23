import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from DataModel.YgoCradModel import YgoCardModel


def image_piechart(total, labels, links, min_zoom=3.5, max_zoom=8):
    """
    Method for creating a pie cards with images in the background of the
    choosen part using matplotlib
    :param total: values for displaying in the plot
    :param label: label of the value classes
    :param links: links to the class images
    """
    fig, axs = plt.subplots(1, figsize=(30, 30))
    axs.axis("equal")
    # modify labels
    tmp = pd.DataFrame(columns=["values", "labels"])
    tmp["values"] = total
    tmp["labels"] = labels
    tmp["ex"] = 0.05
    tmp = tmp[tmp["values"] > 0]
    tmp.sort_values(by="values", ascending=False, inplace=True)
    tmp.reset_index(drop=True, inplace=True)
    total = tmp["values"].to_list()
    labels = tmp["labels"].to_list()
    labs = [f"{lab}\n{tot}" for tot, lab in zip(total, labels)]

    # create pychart
    wedges, texts = axs.pie(
        total,
        startangle=90,
        labels=labs,
        explode=tmp["ex"].to_list(),
        wedgeprops={
            "linewidth": 2,
            "edgecolor": "white",
            "fill": False,
        },
        textprops={
            "fontsize": 55,
            "color": "black",
        },
    )
    # set background images
    sum_total = np.sum(total)
    counter = 0
    card_model = YgoCardModel()
    for i in range(len(labels)):
        category_percentage = total[i] / sum_total
        counter += category_percentage
        if labels[i] == "Rest":
            fn = f"./Deck_Icons/{labels[i]}.png"
            im = plt.imread(fn, format="png")
        else:
            fn = links[labels[i]]
            im = card_model.get_image(fn)

        x = 0.65 * np.cos(2 * np.pi * counter + np.pi / 4)
        y = 0.65 * np.sin(2 * np.pi * counter + np.pi / 4)
        zoom = min_zoom * category_percentage / 0.20
        zoom = np.max([min_zoom, zoom])
        zoom = np.min([max_zoom, zoom])
        img_to_pie(im, wedges[i], xy=(x, y), zoom=zoom)
        wedges[i].set_zorder(10)

    return fig


def img_to_pie(im, wedge, xy, zoom=1, ax=None):
    """
    Method for adding the image to the pie chart
    """
    if ax is None:
        ax = plt.gca()

    path = wedge.get_path()
    patch = PathPatch(path, facecolor="none")
    ax.add_patch(patch)
    imagebox = OffsetImage(im, zoom=zoom, clip_path=patch, zorder=-10)
    ab = AnnotationBbox(imagebox, xy, xycoords="data", pad=0, frameon=False)
    ax.add_artist(ab)
