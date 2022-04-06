import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pyprojroot import here as get_proj_root
import os

px_plt_clrs = px.colors.qualitative.G10

from helpers.helper_fns import load_dict_txt, st_expandable_box

from helpers.animations import (
    run_simulation,
    plot_add_dists,
    get_dists_range,
    get_dists_domains,
    update_plt_rng_dmn,
    plot_figure,
    smooth_zooming_animation,
    create_new_figure,
)

# ddb - drop down box
# sldr - slider
# ckbx - checkbox


def test_plot_rqrs_reframe(st_session_state: st.session_state) -> bool:
    dmn_diff = np.abs(
        st_session_state["plot_domain"] - st_session_state["target_domain"]
    )
    rng_diff = np.abs(st_session_state["plot_range"] - st_session_state["target_range"])

    return np.max([dmn_diff, rng_diff]) > 1e-3


def dashboard_template(dist_cls, dist_name: str, text_file: str):
    if dist_name + "_txt_dict" not in st.session_state:
        proj_root = get_proj_root()
        fil = os.path.join(proj_root, "text_files", text_file)
        st.session_state[dist_name + "_txt_dict"] = load_dict_txt(fil)
    if dist_name + "_dist" not in st.session_state:
        dist = dist_cls(key_root=dist_name, session_state=st.session_state)

        dist.plot_dist_clr = px_plt_clrs[0]
        dist.plot_sim_clr = px_plt_clrs[1]
        dist.plot_cdf_clr = "black"
        dist.plot_mean_clr = px_plt_clrs[3]

        st.session_state[dist_name + "_dist"] = dist

    dist = st.session_state[dist_name + "_dist"]

    st.title(st.session_state[dist_name + "_txt_dict"]["main_title"])

    # * ddb - General information about distribution
    st_expandable_box(
        st.session_state[dist_name + "_txt_dict"],
        "lvl_1_title",
        "lvl_1_text",
        expanded=False,
    )

    # add distribution parameter controls
    with st.expander("Distribution Parameters:", expanded=True):
        dist.create_sliders()

    # add plot option options
    with st.expander("Plot Options:", expanded=False):
        # * ckbx - Select distribution metrics to plot
        cols = st.columns(2)
        with cols[0]:
            plt_dist_mean = st.checkbox(
                "Plot Distribution Expectation",
                value=False,
                key=dist_name + "_plot-mean",
            )
            dist.plot_show_mean = plt_dist_mean
        with cols[1]:
            plt_cdf = st.checkbox(
                "Plot Cumulative Distribution Function",
                value=False,
                key=dist_name + "_plot-CDF",
            )
            dist.plot_show_cdf = plt_cdf
            if dist.dist_type == "discrete":
                plt_cdf_y2 = st.checkbox(
                    "Plot CDF on seperate y-axis?",
                    value=False,
                    disabled=not plt_cdf,
                    key=dist_name + "_plot-CDF",
                )
                dist.plot_cdf_y2 = plt_cdf_y2

        plt_rng_prob_optn = st.selectbox(
            "Display CDF or Range Probability?",
            options=["Neither", "CDF", "Range Probability"],
            key=dist_name + "_plt_rng_prob_optn",
        )

        x_rng = dist.get_plot_domain()
        plt_cdf = False
        if plt_rng_prob_optn != "Neither":
            plt_cdf = True

            # * slider for cdf range
            if plt_rng_prob_optn == "CDF":
                slider_pos = float(x_rng[1])
                slider_title = (
                    "Cumulative Distribution Function: Probability(-infinity --> x)"
                )
            else:  # if plt_rng_prob_optn == "Range Probability"
                slider_pos = (float(x_rng[0]), float(x_rng[1]))
                slider_title = "Range Probability: CDF(end) - CDF(start)"

            if dist.dist_type == "discrete":
                step_size = 0.5
            else:
                step_size = 0.01

            cdf_rng = st.slider(
                slider_title,
                min_value=float(x_rng[0]),
                max_value=float(x_rng[1]),
                step=step_size,
                value=slider_pos,
                key="cdf_rng",
            )

            if plt_rng_prob_optn == "CDF":
                cdf_rng = [cdf_rng]

    # add simulation controls
    with st.expander("Simulation Controls:", expanded=False):
        st.markdown(
            "Simulations will generate random variables based on the user defined distribution. "
            "The number of trials specifies the number of random variables to generate per animated frame. "
            "The animated simulation runs over 5 seconds resulting in 150 frames total. "
            'Generated random variable are "binned" together to create the histogram that is plotted.  \n'
            "Try using 1 trial / frame to see what happens with small sample sets. "
            "How is this different when a large sample set is used (10,000 trials/frame)? "
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

    st.session_state["go_Figure"] = create_new_figure(dist)
    fig = st.session_state["go_Figure"]
    st.session_state["st_plotly_chart"] = st.plotly_chart(fig, use_container_width=True)

    # get required domain and range for the plot
    if "plot_domain" not in st.session_state:
        plot_domain = get_dists_domains([dist], oversize_factor=1)
        plot_range = get_dists_range([dist])
        st.session_state["plot_domain"] = plot_domain
        st.session_state["plot_range"] = plot_range
        st.session_state["target_domain"] = plot_domain
        st.session_state["target_range"] = plot_range
        update_plt_rng_dmn(plot_range, plot_domain, st.session_state)

    plot_add_dists(st.session_state, [dist])
    plot_figure(st.session_state)

    if plt_cdf:
        # * plot bars indicating cdf range
        y_rng = st.session_state["target_range"]
        if len(cdf_rng) == 1:
            line_lbls = ["CDF Range End"]
        else:
            line_lbls = ["Range Start", "Range end"]
        for i in np.arange(len(cdf_rng)):
            fig.add_trace(
                go.Scatter(
                    x=[cdf_rng[i], cdf_rng[i]],
                    y=[y_rng[0], y_rng[1]],
                    line=dict(color=px_plt_clrs[4]),
                    name=line_lbls[i],
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

    plot_figure(st.session_state)

    sim_btn_placeholder.button(
        "Run Simulation",
        on_click=run_simulation,
        args=(st.session_state, [dist], rvs_p_frame),
    )

    # * ddb - General information about distribution
    st_expandable_box(
        st.session_state[dist_name + "_txt_dict"],
        "lvl_2_title",
        "lvl_2_text",
        expanded=True,
    )

    # animate pan/zoom if required
    if test_plot_rqrs_reframe(st.session_state):
        smooth_zooming_animation(
            st_session_state=st.session_state, animation_duration=1.5, dists=[dist]
        )
