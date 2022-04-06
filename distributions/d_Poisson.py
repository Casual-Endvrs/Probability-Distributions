from typing import Optional, List
import streamlit as st
import scipy.stats as stats
import numpy as np
from distributions.discrete_base_class import discrete_base_cls


class Poisson_distribution(discrete_base_cls):
    def __init__(
        self,
        key_root: str,
        session_state,
    ):
        super().__init__(key_root, session_state)

        self.dist_values = np.array(
            [5.0]
        )  # list of the distribution properties - [mean, variance]
        self.plot_dmn = np.array(
            [0, 1]
        )  # specifies the domain required for the distribution to plot over
        self.plot_rng = np.array(
            [0, 1]
        )  # specifies the y-range required for the distribution to plot over

        self.plot_dist_clr = None  # specifies the color of the distribution in the plot
        self.plot_sim_clr = None  # specifies the color of the simulation in the plot

        self.dist_stats = None  # [mean, variance, skew, kurtosis]

        self.x_label = "Random Value"
        self.y_label = "Probability"

        self.reset_sim()
        self._update_dist_pdf()

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution."""
        ref_label = self.key_root + "_" + "Lambda"
        sldr_val = float(self.dist_values[0])

        slider = st.slider(
            "Lambda",
            min_value=0.1,
            max_value=15.0,
            step=0.01,
            value=sldr_val,
            key=ref_label,
            on_change=self._update_class_values,
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

        [dist_start, dist_end] = self.get_plot_domain()

        num_bins = dist_end - dist_start + 1

        self.sim_bins_markers = (
            np.linspace(dist_start, dist_end + 1, num_bins + 1) - 0.5
        )
        self.sim_bins_mid = np.linspace(dist_start, dist_end, num_bins, dtype="int")
        self.sim_bins_cnts = np.zeros(len(self.sim_bins_mid))
        self.sim_total_entries = 0

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

        self.dist = stats.poisson(self.dist_values[0])

        self._calc_dist_stats()

    def _update_class_values(self):  # get the values for each slider
        """Updates the class variable self.dist_values to reflect the current
        slider values.
        """
        for i, key in enumerate(self.slider_keys):
            self.dist_values[i] = self.session_state[key]

        self.reset_sim()
        self._create_dist()

    def _update_plot_dmn(
        self,
    ):  # updates the required plot range based on current distribution parameters
        plt_max = int(self.dist.ppf(0.9999) + 1)
        self.plot_dmn = np.array([-1, plt_max])
