import streamlit as st
import plotly.graph_objects as go
import time
from typing import Optional
import numpy as np


def run_simulation(
    st_figure: st.plotly_chart,
    dists: list,
    num_frame_samples: Optional[int] = 100,
    sim_dur: Optional[int] = 5,
):
    """Runs animation to demonstrate the mathematical distribution prediction 
        versus simulated results of random variables.

    :param st.plotly_chart st_figure: Clean plot to put the results on.
    :param list dists: A list of distributions to plot PMF/PDF and simulation
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
    figure = go.Figure()  # Figure to plot on

    for dist in dists:
        figure = dist.plot_distribution(
            figure
        )  # update the figure with the distribution

    for idx, dist in enumerate(dists):
        figure.add_bar(
            x=dist.sim_bins_mid,
            y=dist.sim_bins_cnts,
            marker_color=None,
            name="Simulation",
            showlegend=True,
        )
        plt_idxs.append(-num_dists + idx)

    while True:
        if time.time() >= frame_tm:
            for idx, dist in enumerate(dists):
                dist = dists[idx]
                dist.simulation_iter(num_frame_samples)
                dist.update_sim_plot_data(figure, plt_idxs[idx])

            st_figure.plotly_chart(figure)

            frame_tm += frame_dur
            frame_num += 1
            if frame_num > ttl_frames:
                break
