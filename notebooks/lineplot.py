""" Utilities for higher level line plots using Bokeh.
    Highly volatile, highly experimental.
"""
import numpy as np
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter
from bokeh.palettes import plasma, Colorblind8


def transform(data, groupby, y, transformation="mean"):
    if transformation == "mean":
        return data.groupby(groupby)[y].mean().to_frame().reset_index()
    elif transformation == "std":
        return data.groupby(groupby)[y].std().to_frame().reset_index()
    elif transformation == "var":
        return data.groupby(groupby)[y].var().to_frame().reset_index()
    else:
        raise ValueError


def set_figure(title, y_axis_scale="linear", y_axis_format=".00"):
    fig = figure(
        title=title,
        plot_height=500,
        plot_width=950,
        y_axis_type=y_axis_scale,
        background_fill_color="#ffffff",
    )
    fig.yaxis.formatter = NumeralTickFormatter(format=y_axis_format)

    if y_axis_scale == "log":
        fig.ygrid.minor_grid_line_color = "#5B5B5B"
        fig.ygrid.minor_grid_line_alpha = 0.1

    return fig


def add_trials(data, x, y, event, trials, event_name, fig, color, aggregate):
    dff = data.loc[data[event] == event_name]

    trial_names = dff[trials].unique()

    alpha = 1 if not aggregate else 0.7
    alpha_stop = 0.2
    alpha_step = (alpha - alpha_stop) / (len(trial_names) - 1)

    lw = 3 if not aggregate else 2

    for trial_name in trial_names:
        df = dff.loc[dff[trials] == trial_name]
        df = df.sort_values(by=[x]).reset_index(drop=True)
        fig.line(
            x=x,
            y=y,
            legend=event_name,
            source=df,
            line_width=lw,
            color=color,
            alpha=alpha,
        )
        alpha -= alpha_step
    return fig


def add_band(x, y, hue_mean, hue_std, hue_name, fig, color):
    hue_std = hue_std.loc[hue_std[y].notnull()].copy()
    hue_std["lower"] = hue_mean[y] - hue_std[y]
    hue_std["upper"] = hue_mean[y] + hue_std[y]

    # Bollinger shading glyph:
    band_x = np.append(hue_std[x].values, hue_std[x].values[::-1])
    band_y = np.append(hue_std["lower"].values, hue_std["upper"].values[::-1])

    fig.patch(x=band_x, y=band_y, legend=hue_name, color=color, alpha=0.3)

    return fig


def lineplot(
    data,
    x,
    y,
    hue=None,
    trials=None,
    aggregate=True,
    legend_pos="top_left",
    y_axis_format="00.00",
    title="Plot",
    y_axis_scale="linear",
):
    # set the figure
    fig = set_figure(
        title, y_axis_format=y_axis_format, y_axis_scale=y_axis_scale
    )

    # get the mean of each event
    y_mean = transform(data, [x, hue], y)

    # get the names of each event we're plotting
    hues = y_mean[hue].unique()
    palette = plasma(len(hues))

    # iterate through events and create a line for each
    for hue_name, color in zip(hues, palette):

        """
        if trials is None:
            # truncate to the slowest trial
            df = data.loc[data[hue] == hue_name]
            min_idx = df.groupby('trial')[x].max().min()
            df = df.loc[data[x] <= min_idx]
        """

        hue_mean = y_mean.loc[y_mean[hue] == hue_name]

        if aggregate:
            fig.line(
                x=x,
                y=y,
                legend=hue_name,
                source=hue_mean,
                line_width=4,
                color=color,
                alpha=1,
            )

        if trials:
            fig = add_trials(
                data, x, y, hue, trials, hue_name, fig, color, aggregate
            )
        else:
            y_std = transform(data, [x, hue], y, transformation="std")
            hue_std = y_std.loc[y_std[hue] == hue_name].copy()
            fig = add_band(x, y, hue_mean, hue_std, hue_name, fig, color)

    # additional settings
    fig.legend.location = legend_pos
    fig.legend.click_policy = "hide"
    return fig


def simple_plot(data, x, y, trials):
    fig = set_figure(title="Simple Plot", y_axis_format="0.0000[00]")

    trial_vals = data[trials].unique()
    trial_names = [f"trial {trial_id}" for trial_id in trial_vals]

    if len(Colorblind8) < len(trial_vals):
        palette = plasma(len(trials))
    else:
        palette = Colorblind8

    for trial, color, legend in zip(trial_vals, palette, trial_names):
        df = data.loc[data[trials] == trial]
        fig.line(
            x=df[x], y=df[y], legend=legend, line_width=4, color=color, alpha=1
        )

    # additional settings
    fig.legend.click_policy = "hide"
    fig.legend.location = "top_left"
    return fig
