import plotly.graph_objects as go


def plot_probabilities(df, markers, amount):
    # Erstelle das Plotly-Objekt
    fig = go.Figure()

    # Füge die Linie für die genaue Wahrscheinlichkeit hinzu
    fig.add_trace(
        go.Scatter(
            x=df["Cards"],
            y=df["Wsk_exact"],
            mode="lines+markers",
            name=f"Genau {amount} Karten",
            line=dict(color="blue"),
            marker=dict(size=6),
        )
    )

    # Füge die Linie für die Mindestwahrscheinlichkeit hinzu
    fig.add_trace(
        go.Scatter(
            x=df["Cards"],
            y=df["Wsk_min"],
            mode="lines+markers",
            name=f"Min. {amount} Karten",
            line=dict(color="green"),
            marker=dict(size=6),
        )
    )

    # Farben für die Marker definieren
    marker_colors = [
        "red",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "yellow",
        "green",
        "black",
        "gray",
    ]

    # Füge die Marker für die gegebenen Schlüssel-Werte-Paare hinzu
    for idx, marker in enumerate(markers):
        key = marker["key"]
        value = marker["value"]

        # Nachschlagen der Wahrscheinlichkeiten für die entsprechende Anzahl von Karten
        wsk_exact = df.loc[df["Cards"] == value, "Wsk_exact"].values[0]
        wsk_min = df.loc[df["Cards"] == value, "Wsk_min"].values[0]

        # Füge einen Kreis für die genaue Wahrscheinlichkeit hinzu (nur einmal in der Legende)
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[wsk_exact],
                mode="markers",
                name=f"Genau {key}",
                marker=dict(
                    size=10,
                    color=marker_colors[idx % len(marker_colors)],
                    opacity=0.8,
                    symbol="circle",
                ),
                showlegend=False,
            )
        )

        # Füge einen Kreis für die Mindestwahrscheinlichkeit hinzu (nur einmal in der Legende)
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[wsk_min],
                mode="markers",
                name=f"Min. {key}",
                marker=dict(
                    size=10,
                    color=marker_colors[idx % len(marker_colors)],
                    opacity=0.8,
                    symbol="circle",
                ),
                showlegend=True,
            )
        )

    # Layout-Optionen festlegen
    fig.update_layout(
        xaxis_title="Anzahl der Karten im Deck",
        yaxis_title="Wahrscheinlichkeit",
        legend=dict(x=1.05, y=1, orientation="v", bordercolor="Black", borderwidth=1),
        hovermode="x unified",
    )

    # Achsenbereich anpassen, falls nötig
    fig.update_yaxes(range=[0, 1.05])  # Da Wahrscheinlichkeiten zwischen 0 und 1 liegen

    return fig
