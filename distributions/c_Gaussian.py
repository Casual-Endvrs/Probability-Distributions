from typing import Optional, Union, List
import streamlit as st
import scipy.stats as stats
import numpy as np
import plotly.graph_objects as go
from distributions.continuous_base_class import continuous_base_cls


class Gaussian_distribution(continuous_base_cls):
    def __init__(
        self,
        key_root: str,
        session_state,
    ):
        super().__init__(key_root, session_state)

        self.dist_values = np.array(
            [0.0, 1.0]
        )  # list of the distribution properties - [mean, standard deviation]
        self.plt_var_width = (
            5  # number of standard deviations to display on either side of the mean
        )

        self.x_label = "Random Value"
        self.y_label = "Probability Density"

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution."""
        self._plt_add_dist_metrics()

        classes = ["Mean", "Standard Deviation"]
        rngs = [[-5.0, 5.0], [0.1, 5.0]]
        for i in np.arange(2):
            slider_text = classes[i]  # "Class: " + str(i + 1)
            ref_label = self.key_root + "_" + str(i)

            sldr_min = rngs[i][0]
            sldr_mx = rngs[i][1]
            sldr_val = float(self.dist_values[i])

            slider = st.slider(
                slider_text,
                min_value=sldr_min,
                max_value=sldr_mx,
                step=0.01,
                value=sldr_val,
                key=ref_label,
                on_change=self._update_class_values,
            )
            if ref_label not in self.slider_keys:
                self.slider_keys.append(ref_label)

    #! Internal Functions

    def _create_dist(
        self,
        mean: Optional[float] = None,
        standard_deviation: Optional[int] = None,
    ):  # creates the scipy.stats distribution
        """Creates a new instance of a scipy.stats binomial distribution which
            is stored in the class variable self.dist. The distribution can be
            created using specified values for the success rate and number of
            trials or using the classes currently values in self.dist_values.

        :param Optional[float] success_rate (optional): Success rate of the
            Bernoulli trials. Must be between 0 & 1. Defaults to None.
        :param Optional[float] num_trials (optional): Number of Bernoulli
            trials. Must be an integer value greater than 0. Defaults to None.
        """
        if mean is not None:
            self.dist_values[0] = mean
            self._update_sliders()
        if standard_deviation is not None:
            self.dist_values[1] = standard_deviation
            self._update_sliders()

        self.dist = stats.norm(self.dist_values[0], self.dist_values[1])

        self._calc_dist_stats()

    def _update_plot_rng(
        self,
    ):  # updates the required plot range based on current distribution parameters
        plt_min = self.dist_values[0] - self.plt_var_width * self.dist_values[1]
        plt_max = self.dist_values[0] + self.plt_var_width * self.dist_values[1]
        self.plot_rng = np.array([plt_min, plt_max])
