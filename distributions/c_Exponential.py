from typing import Optional
import streamlit as st
import scipy.stats as stats
import numpy as np
from distributions.continuous_base_class import continuous_base_cls


class Exponential_distribution(continuous_base_cls):
    def __init__(
        self,
        key_root: str,
        session_state,
    ):
        super().__init__(key_root, session_state)

        self.dist_values = np.array(
            [1.0]
        )  # list of the distribution properties - [mean, variance]

        self.x_label = ""
        self.y_label = "Probability Density"

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution."""
        ref_label = self.key_root + "_" + "Lambda"

        sldr_val = float(self.dist_values[0])

        slider = st.slider(
            "Lambda",
            min_value=0.1,
            max_value=5.0,
            step=0.01,
            value=sldr_val,
            key=ref_label,
            on_change=self._update_class_values,
        )
        if ref_label not in self.slider_keys:
            self.slider_keys.append(ref_label)

    #! Internal Functions

    def _create_dist(
        self, decay: Optional[float] = None
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
        if decay is not None:
            self.dist_values[0] = decay
            self._update_sliders()

        self.dist = stats.expon(0, self.dist_values[0])

        self._calc_dist_stats()

    def _update_plot_dmn(
        self,
    ):  # updates the required plot range based on current distribution parameters
        plt_min = 0
        plt_max = self.dist.ppf(0.999)
        self.plot_dmn = np.array([plt_min, plt_max])
