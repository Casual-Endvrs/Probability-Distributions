from typing import Optional, Union
import streamlit as st
import scipy.stats as stats
import numpy as np
import plotly.graph_objects as go


class Beta_distribution:
    def __init__(
        self, key_root: str, session_state,
    ):
        self.initialization_errors = []

        self.key_root = key_root
        self.session_state = session_state

        self.dist = None  # scipy.stats distribution
        self.dist_pdf = (
            None  # an array of the distributions PMF/PDF based on sim_bins_mid
        )
        self.dist_type = "continuous"  # specifies "discrete" vs "continuous"
        self.slider_keys = []  # keys to access slider values in st.session_state
        self.sim_num_bins = None  # total number of bins for simulation
        self.bin_width = None  # width of the simulation bin
        self.sim_bins_markers = None  # edge limits for each bin edge for creating historgrams and CDF calculations
        self.sim_bins_mid = None  # mid pont for plotting
        self.sim_bins_cnts = None  # binned data from the simulation
        self.sim_total_entries = 0  # total number of simulation entries
        self.dist_values = np.array(
            [2.0, 5.0]
        )  # list of the distribution properties - [mean, variance]
        self.plot_rng = None  # list of two values indicating the required range for this distribution to be plotted over

        self.plot_dist_clr = None  # specifies the color of the distribution in the plot
        self.plot_sim_clr = None  # specifies the color of the simulation in the plot

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution.
        """
        classes = ["Alpha", "Beta"]
        rngs = [[0.01, 5.0], [0.01, 5.0]]
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

    def cdf(self, x: float) -> float:  # cdf of the distribution from -infinity to x
        """Returns the probability of obtaining a random value between 
            -infinity and x for the mathematical definition of the 
            distribution.

        :param float x: Upper limit for value range.
        :float: Probability for the provided range.
        """
        return self.dist.cdf(x)

    def range_probability(
        self, x_0: Union[list, np.ndarray, tuple, float], x_1: Optional[float] = None
    ) -> float:  # cdf of the distribution over the range x_0 --> x_1
        """Returns a single float value for the probability of obtaining a 
            value between x_0 and x_1 for the ideal distribution, i.e. based 
            on the mathematical definition of the distribution.

        :param Union[list, np.ndarray, tuple, float] x_0: This can be either 
            the initial value of the range of interest or a list of two values 
            specifying the initial and final values of the range.
        :param Optional[float] x_1 (optional): The final value for the range
            of interest. If left as None, then x_0 must be a list specifying 
            the initial and final points of the range. Defaults to None.
        :float: Probability of obtaining a value between x_0 & x_1.
        """
        if isinstance(x_0, (list, np.ndarray, tuple)):
            x_1 = x_0[1]
            x_0 = x_0[0]

        cdf_0 = self.cdf(x_0)
        cdf_1 = self.cdf(x_1)

        return cdf_1 - cdf_0

    def sim_cdf(self, x: float) -> float:  # cdf of the simulation from -infinity to x
        """Returns the probability of obtaining a random value between 
            -infinity and x for the simulation results.

        :param float x: Upper limit for value range.
        :float: Probability for the provided range.
        """
        bins = self.sim_bins_mid <= x

        return np.sum(self.sim_bins_cnts[bins]) / self.sim_total_entries

    def sim_range_probability(
        self, x_0: Union[list, np.ndarray, tuple, float], x_1: Optional[float] = None
    ) -> float:  # cdf of the simulation results
        """Returns a single float value for the probability of obtaining a 
            value between x_0 and x_1 for the simulation results.

        :param Union[list, np.ndarray, tuple, float] x_0: This can be either 
            the initial value of the range of interest or a list of two values 
            specifying the initial and final values of the range.
        :param Optional[float] x_1 (optional): The final value for the range
            of interest. If left as None, then x_0 must be a list specifying 
            the initial and final points of the range. Defaults to None.
        :float: Probability of obtaining a value between x_0 & x_1.
        """
        # safety measure incase the simulation has not been run yet
        if self.sim_total_entries == 0:
            return 0

        # get the indices of the bins for each point of the provided range
        if isinstance(x_0, (list, np.ndarray, tuple)):
            x_1 = x_0[1]
            x_0 = x_0[0]

        cdf_0 = self.sim_cdf(x_0)
        cdf_1 = self.sim_cdf(x_1)

        # return sum of cdf range, normalized using total number of sim entries
        return cdf_1 - cdf_0

    def get_plot_range(
        self,
    ) -> np.ndarray:  # calculate the range required for the distribution to be plotted
        """Returns the range that is required to plot the distribution.

        :np.ndarray: An array of two values indicating the range of the plot.
        """
        self._update_plot_rng()
        return self.plot_rng

    def get_plot_domain(
        self,
    ) -> np.ndarray:  # returns the max y value required to plot this data
        """Returns the domain of plots generated by this function. This will 
            consider both the mathematical definition of the function itself as 
            well as any simulation results.

        :np.ndarray: An array of two values indicating the domain of the plot.
        """
        y_maxs = [self.dist_pdf]
        if self.sim_total_entries > 0:
            y_maxs.append(self._get_sim_bins_normalized())
        return np.array([0, np.max(y_maxs)])

    def simulation_iter(self, num_samples: int):  # perform a simulation step
        """Performs a single run of the simulation and updates all required
            internal parameters.

        :param int num_samples: Number of random variables to generate for this
            simulation run.
        """
        # get a total of num_samples of random values
        random_vals = self.dist.rvs(num_samples)

        # bin the random values
        binned_rvs, _ = self._bin_sim_rvs(random_vals)

        # add the new random value bin counts to current tally
        self.sim_bins_cnts += binned_rvs
        self.sim_total_entries += num_samples

    def reset_sim(
        self, bin_rng: Optional[list[float]] = None
    ):  # updates the distribution to reflect current values and resets all the relevant variables for the simulation
        """Resets internal variables to prepare for a new simulation run.

        :param Optional[list[float]] bin_rng (optional): A list of values 
            indicating the limits of the bins used to split up the simulation 
            results. If N bin limits are provided, N-1 bins will be made. 
            Defaults to None.
        """
        self._create_dist()
        if self.sim_num_bins is None:
            self.sim_num_bins = 100

        [dist_start, dist_end] = self.get_plot_range()

        self.bin_width = (dist_end - dist_start) / self.sim_num_bins
        self.sim_bins_markers = np.linspace(dist_start, dist_end, self.sim_num_bins + 1)
        self.sim_bins_mid = self.sim_bins_markers[:-1] + self.bin_width / 2
        self.sim_bins_cnts = np.zeros(int(self.sim_num_bins))
        self.sim_total_entries = 0

    def plot_distribution(
        self, figure: go.Figure
    ) -> go.Figure:  # plot the distributions pmf / pdf
        """Plots the distribution and simulation results (if available) on the
            provided figure. This should only be performed on a figure which 
            does not contain a plot of the distribution or the simulation 
            results as this will add new plots and not update previous values.

        :param go.Figure figure: A plotly.graph_object.Figure.
        :go.Figure: The Figure with the distribution and simulation results 
            plotted.
        """
        # update the PMF/PDF values of the distribution
        self._update_dist_pdf()

        # plot bar graph for the distribution
        figure.add_scatter(
            x=self.sim_bins_mid,
            y=self.dist_pdf,
            name="Binomial Expectation",
            marker_color=self.plot_dist_clr,
        )

        if self.sim_total_entries > 0:
            sim_bin_cnts = self._get_sim_bins_normalized()
            figure.add_bar(
                x=self.sim_bins_mid,
                y=sim_bin_cnts,
                marker_color=self.plot_sim_clr,
                name="Binomial Simulation",
                showlegend=True,
            )

        return figure

    def update_sim_plot_data(self, figure: go.Figure, data_idx: int) -> go.Figure:
        """Updates the simulation data of the plotted figure. Requires the 
            index of the data to be updated.

        :param go.Figure figure: A plotly.graph_object.Figure that is to be
            updated.
        :param int data_idx: The index of the data to be updated.
        :go.Figure: The updated Figure.
        """
        sim_vals = self._get_sim_bins_normalized()
        figure["data"][data_idx]["y"] = sim_vals

        return figure

    #! Internal Functions

    def _create_dist(
        self, mean: Optional[float] = None, width: Optional[int] = None,
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
        if width is not None:
            self.dist_values[1] = width
            self._update_sliders()

        self.dist = stats.beta(a=self.dist_values[0], b=self.dist_values[1])

    def _update_sliders(
        self,
    ):  # updates the slider values to the current class ratio values
        """Updates the slider values to match the class values. This is often
            done after the class ratios have been normalized.
        """
        for i, key in enumerate(self.slider_keys):
            self.session_state[key] = self.dist_values[i]

    def _update_class_values(self):  # get the values for each slider
        """Updates the class variable self.dist_values to reflect the current
            slider values.
        """
        for i, key in enumerate(self.slider_keys):
            val = self.session_state[key]
            self.dist_values[i] = val

        self.reset_sim()
        self._create_dist()

    def _get_sim_bins_normalized(
        self,
    ) -> Optional[
        list
    ]:  # returns the current sim_bins counts normalized to match the distribution curve
        """If a simulation has been run, returns the number of entries in each 
            bin, normalized by the total number of random values used in the 
            simulation.

        :Optional[list]: If a simulation has been run a list of float values 
            will be returned. If no simulation has been run, None is returned.
        """
        if self.sim_total_entries == 0:
            return None

        return self.sim_bins_cnts / self.sim_total_entries / self.bin_width

    def _bin_sim_rvs(
        self, random_vals: np.ndarray
    ) -> np.ndarray:  # bins the random values from the simulation
        """Bins data from the simulation.

        :param np.ndarray random_vals: An array containing the simulation 
            values.
        :np.ndarray: The count of the number of entries per bin.
        """
        #! requires a test to ensure bins are defined
        binned_rvs = np.histogram(random_vals, bins=self.sim_bins_markers)
        return binned_rvs

    def _update_dist_pdf(self,):  # returns the PMF/PDF of the distribution
        """Returns the PMF/PDF values of the distribution calculated at the 
            middle of each bin used for the simulation results.
        """
        self._create_dist()
        self.dist_pdf = self.dist.pdf(self.sim_bins_mid)

    def _update_plot_rng(
        self,
    ):  # updates the required plot range based on current distribution parameters
        plt_min = -0.05
        plt_max = 1.05
        self.plot_rng = np.array([plt_min, plt_max])

