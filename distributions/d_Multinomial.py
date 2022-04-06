from typing import Optional, Union, List
import streamlit as st
import numpy as np
from distributions.discrete_base_class import discrete_base_cls


class Multinomial_distribution(discrete_base_cls):
    def __init__(
        self,
        key_root: str,
        session_state,
        norm_method: Optional[str] = None,
    ):
        super().__init__(key_root, session_state, norm_method)

        self.num_classes = 6  # number of classes
        self.sim_bin_width = 1  # width of the simulation bin #! currently unused.
        self.sim_bins_cnts = np.zeros(
            self.num_classes
        )  # binned data from the simulation
        self.dist_values = (
            np.ones(self.num_classes) / self.num_classes
        )  # list of the distribution properties

        self.plot_dmn = np.array(
            [-1, 2]
        )  # specifies the domain required for the distribution to plot over
        self.plot_rng = np.array(
            [0, 1]
        )  # specifies the y-range required for the distribution to plot over

        #! Note: The mean value is calculated but variance, skew & kurtosis are set to zero, 0
        self.dist_stats = None  # [mean, variance, skew, kurtosis]

        self.x_label = "Random Value"
        self.y_label = "Probability"

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution."""
        for i in np.arange(self.num_classes):
            slider_text = "Class: " + str(i + 1)
            ref_label = self.key_root + "_" + str(i)
            slider = st.slider(
                slider_text,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=1 / self.num_classes,
                key=ref_label,
                on_change=self._normalize_class_ratios,
                args=(i,),
            )
            if ref_label not in self.slider_keys:
                self.slider_keys.append(ref_label)

    def cdf(
        self, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:  # cdf of the distribution over the range x_0 --> x_1
        """Returns the probability of obtaining a random value between
            -infinity and x for the mathematical definition of the
            distribution.

        :param float x: Upper limit for value range.
        :float: Probability for the provided range.
        """
        if isinstance(x, (list, np.ndarray)):
            xs = x
            cdf = []
            for x in xs:
                idxs = self.sim_bins_mid <= x
                cdf.append(np.sum(self.dist_values[idxs]))
            cdf = np.array(cdf)
        else:
            idxs = self.sim_bins_mid <= x
            cdf = np.sum(self.dist_values[idxs])

        return cdf

    def simulation_iter(self, num_samples: int):  # perform a simulation step
        """Performs a single run of the simulation and updates all required
            internal parameters.

        :param int num_samples: Number of random variables to generate for this
            simulation run.
        """
        # get a total of num_samples of random values
        random_vals = np.random.multinomial(num_samples, self.dist_values)

        # add the new random value bin counts to current tally
        self.sim_bins_cnts += random_vals
        self.sim_total_entries += num_samples

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
        self.sim_bins_markers = np.arange(self.num_classes + 1) - 0.5
        self.sim_bins_mid = np.arange(self.num_classes) + 1
        self.sim_bins_cnts = np.zeros(self.num_classes)
        self.sim_total_entries = 0

    #! Internal Functions

    #! set this to take a list
    def _create_dist(  #! add documentation
        self,
        class_probabilities: Optional[List[float]] = None,
    ):  # creates the scipy.stats distribution
        """
        #! add documentation
        """
        if class_probabilities is not None:
            self.dist_values = np.array(class_probabilities)
            self.num_classes = len(class_probabilities)

        self._calc_dist_stats()

    def _update_dist_pdf(
        self,
    ):  # returns the PMF/PDF of the distribution
        """Returns the PMF/PDF values of the distribution calculated at the
        middle of each bin used for the simulation results.
        """
        self._create_dist()
        self.dist_pdf = self.dist_values
        self.dist_pdf_max = np.max(self.dist_pdf)
        self._update_dist_cdf()

    def _update_plot_dmn(
        self,
    ):  # updates the required plot range based on current distribution parameters
        self.plot_dmn = np.array([0, self.num_classes + 1])

    def _calc_dist_stats(self):
        dist_mean = np.sum(self.dist_values * np.arange(1, self.num_classes + 1))

        self.dist_stats = [dist_mean, 0, 0, 0]
