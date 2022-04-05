import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Optional, List
import numpy as np
import math


def create_new_figure(dist=None):
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)

    if dist is not None:
        x_label = None
        y_label = None
        if hasattr(dist, "x_label"):
            x_label = dist.x_label
        if hasattr(dist, "y_label"):
            y_label = dist.y_label
        figure.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        if hasattr(dist, "x_tick_markers"):
            figure.update_layout(
                xaxis=dict(
                    tickvals=dist.x_tick_markers[0], ticktext=dist.x_tick_markers[1]
                )
            )
        if dist.dist_type == "discrete":
            figure.update_layout(xaxis_showgrid=True, yaxis_showgrid=False)

    return figure


def plot_add_dists(st_session_state: st.session_state, dists: List):
    figure = st_session_state["go_Figure"]
    for dist in dists:
        figure = dist.plot_distribution(figure)

    set_target_rng_dmn(st_session_state, dists)


def plot_figure(st_session_state: st.session_state):
    plot_use_ssn_stt_rng_dmn(st_session_state)
    st_session_state["st_plotly_chart"].plotly_chart(st_session_state["go_Figure"])


def run_simulation(  #! changed input variables
    st_session_state: st.session_state,
    dists: List,
    num_frame_samples: Optional[int] = 100,
    sim_dur: Optional[int] = 5,
):
    """Runs animation to demonstrate the mathematical distribution prediction
        versus simulated results of random variables.

    :param st.plotly_chart st_figure: Clean plot to put the results on.
    :param List dists: A list of distributions to plot PMF/PDF and simulation
        results.
    :param Optional[int] num_frame_samples (optional): Number of random
        random variables to generate for each frame of the simulation. Defaults
        to 100.
    :param Optional[int] sim_dur (optional): Duration of time, in seconds, for
        the simulation to run over. Defaults to 5.
    """
    plt_idxs = []
    num_dists = len(dists)
    for dist in dists:
        dist.reset_sim()  # reset the values for the simulation
    sim_dur = 5  # duration of animation in seconds
    frames_per_sec = 30  # frames per second for the animation
    frame_dur = 1 / frames_per_sec  # duration in seconds for each frame
    ttl_frames = (
        sim_dur * frames_per_sec
    )  # total number of frames for the animation to run over
    frame_tm = time.time()  # time to display next frame
    frame_num = 1  # current frame number
    figure = create_new_figure(dists[0])  # Figure to plot on
    st_session_state["go_Figure"] = figure

    # update the figure with the distribution
    plot_add_dists(st_session_state, dists)

    for idx, dist in enumerate(dists):
        figure.add_bar(
            x=dist.sim_bins_mid,
            y=dist.sim_bins_cnts,
            marker_color=dist.plot_sim_clr,
            name="Simulation",
            showlegend=True,
        )
        #! I expect this to be wrong, plt_idxs may need to be reversed [::-1]
        plt_idxs.append(-num_dists + idx)

    while True:
        if time.time() >= frame_tm:
            for idx, dist in enumerate(dists):
                dist.simulation_iter(num_frame_samples)
                dist.update_sim_plot_data(figure, plt_idxs[idx])

            smooth_zoom_step(st_session_state, 30, dists=dists)

            frame_tm += frame_dur
            frame_num += 1
            if frame_num > ttl_frames:
                break

    set_target_rng_dmn(st_session_state, dists=dists)
    smooth_zooming_animation(
        st_session_state=st_session_state, animation_duration=0.5, dists=dists
    )


def smooth_zoom_step(
    st_session_state: st.session_state,
    step_factor: int = 30,
    dists: Optional[List] = None,
):
    # get required plot variables
    st_plotly_chart = st_session_state["st_plotly_chart"]
    figure = st_session_state["go_Figure"]

    # get initial domain and range for the current plot
    range_start = np.array(st_session_state["plot_range"])
    domain_start = np.array(st_session_state["plot_domain"])

    # get final domain and range for the plot
    range_stop = get_range(figure, dists=dists, oversize_factor=1)
    domain_stop = get_domain(figure, dists=dists, oversize_factor=1.075)

    # create step sizes
    rng_ss = (
        np.array([range_stop[0] - range_start[0], range_stop[1] - range_start[1]])
        / step_factor
    )
    dmn_ss = (
        np.array([domain_stop[0] - domain_start[0], domain_stop[1] - domain_start[1]])
        / step_factor
    )

    # update range and domain
    range_start += rng_ss
    domain_start += dmn_ss

    # update plot
    update_plt_rng_dmn(range_start, domain_start, st_session_state)
    st_plotly_chart.plotly_chart(figure)


def smooth_zooming_animation(
    st_session_state: st.session_state,
    animation_duration: float = 1,
    dists: Optional[List] = None,
):
    # get required plot variables
    st_plotly_chart = st_session_state["st_plotly_chart"]
    figure = st_session_state["go_Figure"]

    # get initial domain and range for the current plot
    range_start = np.array(st_session_state["plot_range"])
    domain_start = np.array(st_session_state["plot_domain"])

    # get final domain and range for the plot
    range_stop = get_range(figure, dists=dists, oversize_factor=1)
    domain_stop = get_domain(figure, dists=dists, oversize_factor=1.075)

    # create step sizes
    rng_diff = np.array(
        [range_stop[0] - range_start[0], range_stop[1] - range_start[1]]
    )
    dmn_diff = np.array(
        [domain_stop[0] - domain_start[0], domain_stop[1] - domain_start[1]]
    )

    num_steps = int(30 * animation_duration)
    frame_time = animation_duration / num_steps

    nxt_frame_tm = time.time()
    for i in np.arange(num_steps):
        nxt_frame_tm += frame_time
        sigmoid_x = sigmoid(14 * i / num_steps - 7)
        plt_rng = range_start + rng_diff * sigmoid_x
        plt_dmn = domain_start + dmn_diff * sigmoid_x
        update_plt_rng_dmn(plt_rng, plt_dmn, st_session_state)
        st_plotly_chart.plotly_chart(figure)
        delay_until_time(nxt_frame_tm)

    update_plt_rng_dmn(range_stop, domain_stop, st_session_state)
    st_plotly_chart.plotly_chart(figure)


def set_target_rng_dmn(st_session_state: st.session_state, dists: list):
    rng = get_dists_range(dists, oversize_factor=1)
    dmn = get_dists_domains(dists, oversize_factor=1.075)
    st_session_state["target_range"] = rng
    st_session_state["target_domain"] = dmn


def plot_use_ssn_stt_rng_dmn(st_session_state: st.session_state):
    x_range = st_session_state["plot_range"]
    y_domain = st_session_state["plot_domain"]
    update_plt_rng_dmn(x_range, y_domain, st_session_state)
    st_session_state["st_plotly_chart"].plotly_chart(st_session_state["go_Figure"])


def get_range(
    figure: Optional[go.Figure] = None,
    dists: Optional[List] = None,
    oversize_factor: Optional[float] = 1.075,
) -> np.ndarray:
    if dists is not None:
        return get_dists_range(dists, oversize_factor)
    elif figure is not None:
        return get_plot_range(figure, oversize_factor)
    else:
        return np.array([0, 1])


def get_domain(
    figure: Optional[go.Figure] = None,
    dists: Optional[List] = None,
    oversize_factor: Optional[float] = 1.075,
) -> np.ndarray:
    if dists is not None:
        return get_dists_domains(dists, oversize_factor)
    elif figure is not None:
        return get_plot_domain(figure, oversize_factor)
    else:
        return np.array([0, 1])


def get_plot_range(figure: go.Figure, oversize_factor: float = 1.075) -> np.ndarray:
    fig_data = figure["data"]
    y_data = []

    for i in np.arange(len(fig_data)):
        y_data.extend(fig_data[i]["x"])

    return np.array([np.min(y_data), np.max(y_data)]) * oversize_factor


def get_plot_domain(figure: go.Figure, oversize_factor: float = 1.075) -> np.ndarray:
    fig_data = figure["data"]
    x_data = []

    for i in np.arange(len(fig_data)):
        x_data.extend(fig_data[i]["y"])

    return np.array([np.min(x_data), np.max(x_data)]) * oversize_factor


def update_plt_rng_dmn(
    x_range: np.ndarray,
    y_domain: np.ndarray,
    st_session_state: st.session_state,
):
    st_session_state["plot_range"] = x_range
    st_session_state["plot_domain"] = y_domain
    st_session_state["go_Figure"].update_xaxes(range=x_range)
    st_session_state["go_Figure"].update_yaxes(range=y_domain, secondary_y=False)


def delay_until_time(delay_until: time.time):
    while True:
        if time.time() >= delay_until:
            break


def get_dists_range(dists: List, oversize_factor: float = 1.075) -> np.ndarray:
    ranges = []

    for dist in dists:
        ranges.extend(dist.get_plot_range())

    return np.array([np.min(ranges), np.max(ranges)], dtype=np.float) * oversize_factor


def get_dists_domains(dists: List, oversize_factor: float = 1.075) -> np.ndarray:
    domains = []

    for dist in dists:
        domains.extend(dist.get_plot_domain())

    return (
        np.array([np.min(domains), np.max(domains)], dtype=np.float) * oversize_factor
    )


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
