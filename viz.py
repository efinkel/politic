import numpy as np
import pandas as pd
import matplotlib
import plotly
import plotly.graph_objects as go

def get_word_colormap(keyword_list, colormap):
    cm = matplotlib.colormaps[colormap]
    colors = list(cm.colors)
    np.random.shuffle(colors)

    word_color_map = {word: plotly.colors.label_rgb(c) for word, c in zip(keyword_list, colors)}
    return word_color_map


def get_word_y_vals(keyword_list):
    return {word: (1 + i * 0.1) for i, word in enumerate(keyword_list)}


def plot_keyphrase_raster(keyphrases):
    keyphrase_df = pd.DataFrame(keyphrases)
    keyword_counts = keyphrase_df['word'].value_counts()
    frequent_kw = list(keyword_counts[keyword_counts > 2].index)[::-1]
    frequent_keyphrase_df = keyphrase_df.query(f'word in {frequent_kw}')

    word_color_map = get_word_colormap(frequent_kw, 'tab20')
    word_y_map = get_word_y_vals(frequent_kw)

    color_list = frequent_keyphrase_df['word'].map(word_color_map).values
    y_list = frequent_keyphrase_df['word'].map(word_y_map).values

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode='markers', x=frequent_keyphrase_df['start'],
            y=y_list,
            marker={
                'symbol': 'line-ns',
                'size': 12,
                'line': {'width': 3, 'color': color_list, }
            },
        ))

    fig.update_layout(
        width=1200,
        height=300,
        margin=dict(l=50, r=0, b=50, t=0, pad=4),
        template="ggplot2",
    )

    # import pdb; pdb.set_trace()
    fig.update_yaxes(tickvals=list(word_y_map.values()), ticktext=frequent_kw)

    return fig