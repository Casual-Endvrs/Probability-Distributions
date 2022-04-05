from typing import Optional, Union, List
import streamlit as st
import scipy.stats as stats
import numpy as np
import plotly.graph_objects as go
from distributions.discrete_base_class import discrete_base_cls


class Bernoulli_distribution(discrete_base_cls):
    def __init__(
        self,
        key_root: str,
        session_state,
        norm_method: Optional[str] = None,
    ):
        super().__init__(key_root, session_state, norm_method)

        self.dist_values = np.array([0.5, 0.5])  # list of the distribution properties
        self.plot_rng = np.array(
            [-1, 2]
        )  # list of two values indicating the required range for this distribution to be plotted over

        self.x_label = "Random Outcome"
        self.y_label = "Probability"
        self.x_tick_markers = [[0, 1], ["0 - Failure", "1 - Success"]]

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution."""
        self._plt_add_dist_metrics()

        classes = ["Success Rate", "Failure Rate"]
        for i in np.arange(2):
            slider_text = classes[i]  # "Class: " + str(i + 1)
            ref_label = self.key_root + "_" + str(1 - i)
            st.slider(
                slider_text,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=0.5,
                key=ref_label,
                on_change=self._normalize_class_ratios,
                args=(1 - i,),
            )
            if ref_label not in self.slider_keys:
                self.slider_keys.append(ref_label)

    def reset_sim(
        self, bin_rng: Optional[List[float]] = None
    ):  # updates the distribution to reflect current values and resets all the relevant variables for the simulation
        """Resets internal variables to prepare for a new simulation run.

        :param Optional[List[float]] bin_rng (optional): A list of values
            indicating the limits of the bins used to split up the simulation
            results. If N bin limits are provided, N-1 bins will be made.
            Defaults to None.
        """
        self._create_dist()
        self.sim_bin_width = 1
        self.sim_bins_markers = np.array([-0.5, 0.5, 1.5])
        self.sim_bins_mid = np.array([0, 1])
        self.sim_bins_cnts = np.array([0, 0])
        self.sim_total_entries = 0

    #! Internal Functions

    def _create_dist(
        self,
        success_rate: Optional[float] = None,
    ):  # creates the scipy.stats distribution
        """Creates an instance of a scipy.stats Bernoulli distribution and
            stores the result in the self.dist variable.

        :param Optional[float] success_rate (optional): A value between 0 & 1
            which specifies the success rate of Bernoulli trials. Defaults to
            None.
        """
        if success_rate is not None:
            self.dist_values[0] = 1 - success_rate
            self.dist_values[1] = success_rate
            self._update_sliders()
        self.dist = stats.bernoulli(self.dist_values[1])

        self._calc_dist_stats()

    def _update_sliders(
        self,
    ):  # updates the slider values to the current class ratio values
        """Updates the slider values to match the class values. This is often
        done after the class ratios have been normalized.
        """
        for i, key in enumerate(self.slider_keys):
            self.session_state[key] = self.dist_values[1 - i]

    def _update_plot_rng(
        self,
    ):  # updates the required plot range based on current distribution parameters
        self.plot_rng = [-1, 2]
