import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pyprojroot import here as get_proj_root
import os

from helpers.helper_fns import load_dict_txt, st_expandable_box

from distributions.d_Binomial import Binomial_distribution

from helpers.animations import (
    run_simulation,
    plot_add_dists,
    get_dists_range,
    get_dists_domains,
    update_plt_rng_dmn,
    plot_figure,
    smooth_zooming_animation,
    create_new_figure
)

# ddb - drop down box
# sldr - slider


def test_plot_rqrs_reframe(st_state_session: st.session_state):
    rng_diff = np.abs(st.session_state["plot_range"] - st.session_state["target_range"])
    dmn_diff = np.abs(
        st.session_state["plot_domain"] - st.session_state["target_domain"]
    )

    return np.max([rng_diff, dmn_diff]) > 1e-3


def app():
    if "Binomial_txt_dict" not in st.session_state:
        proj_root = get_proj_root()
        fil = os.path.join(proj_root, "text_files", "d_Binomial.txt")
        st.session_state["Binomial_txt_dict"] = load_dict_txt(fil)
    if "Binomial_dist" not in st.session_state:
        st.session_state["Binomial_dist"] = Binomial_distribution(
            key_root="Binomial_dist", session_state=st.session_state
        )
    dist = st.session_state["Binomial_dist"]

    st.header(st.session_state["Binomial_txt_dict"]["main_title"])

    # * ddb - General information about distribution
    st_expandable_box(
        st.session_state["Binomial_txt_dict"], "lvl_0_title", "lvl_0_text"
    )

    dist.create_sliders()
    x_rng = dist.get_plot_range()

    # * slider for cdf range
    cdf_rng = st.slider(
        "CDF - Cumulative Density Function",
        min_value=float(x_rng[0]),
        max_value=float(x_rng[1]),
        step=0.01,
        value=(float(x_rng[0]), float(x_rng[1])),
        key="cdf_rng",
    )

    cols = st.columns(3)
    with cols[0]:
        # * ddb: number of trials per frame
        rvs_p_frame = st.selectbox(
            "Number of Trials per Frame",
            options=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
            index=5,
            key="sim_rvs_p_frame",
        )
    with cols[1]:
        st.markdown("#")  # spacing
        ttl_rvs = rvs_p_frame * 30 * 5
        st.write(f"Total number of simulated random variables: {ttl_rvs:,}")
    with cols[2]:
        st.markdown("#")  # spacing
        sim_btn_placeholder = st.empty()

    #! working

    st.session_state["go_Figure"] = create_new_figure()
    fig = st.session_state["go_Figure"]
    st.session_state["st_plotly_chart"] = st.plotly_chart(fig, use_container_width=True)

    if "plot_range" not in st.session_state:
        plot_range = get_dists_range([dist], oversize_factor=1)
        plot_domain = get_dists_domains([dist])
        st.session_state["plot_range"] = plot_range
        st.session_state["plot_domain"] = plot_domain
        st.session_state["target_range"] = plot_range
        st.session_state["target_domain"] = plot_domain
        update_plt_rng_dmn(plot_range, plot_domain, st.session_state)

    plot_add_dists(st.session_state, [dist])
    plot_figure(st.session_state)

    # * plot bars indicating cdf range
    y_rng = st.session_state["target_domain"]
    cdf_line_lbls = ["CDF Range Start", "CDF Range End"]
    for i in np.arange(2):
        fig.add_trace(
            go.Scatter(
                x=[cdf_rng[i], cdf_rng[i]],
                y=[y_rng[0], y_rng[1]],
                line=dict(color="red"),
                name=cdf_line_lbls[i],
            )
        )

    # * get cdf values
    dist_cdf = dist.range_probability(cdf_rng)

    # * print cdf value on the plot
    cdf_plt_txt = [f"<b>CDF (Distribution) = {dist_cdf:.3f}</b>"]
    if dist.sim_total_entries > 0:
        sim_cdf = dist.sim_range_probability(cdf_rng)
        cdf_plt_txt.append(f"<b>CDF (Simulation) = {sim_cdf:.3f}</b>")

    for i, txt in enumerate(cdf_plt_txt):
        fig.add_annotation(
            dict(
                font=dict(color="black", size=17),
                x=x_rng[0] + 0.05 * np.mean(np.abs(x_rng)),
                y=y_rng[1] * (0.9 - i / 10),
                showarrow=False,
                textangle=0,
                xanchor="left",
                text=txt,
            )
        )

    st.session_state["go_Figure"] = fig

    plot_figure(st.session_state)

    sim_btn_placeholder.button(
        "Run Simulation",
        on_click=run_simulation,
        args=(st.session_state, [dist], rvs_p_frame),
    )

    # * ddb - General information about distribution
    st_expandable_box(
        st.session_state["Binomial_txt_dict"], "lvl_1_title", "lvl_1_text"
    )

    if test_plot_rqrs_reframe(st.session_state):
        smooth_zooming_animation(
            st_session_state=st.session_state, animation_duration=2,
        )
