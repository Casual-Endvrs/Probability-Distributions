from typing import Optional, Union, List
import streamlit as st
import numpy as np
import plotly.graph_objects as go


class Multinomial_distribution:
    def __init__(
        self, key_root: str, session_state, norm_method: Optional[str] = None,
    ):
        self.initialization_errors = []

        self.key_root = key_root
        self.session_state = session_state

        if norm_method is None:
            self.normalization_method = "normalize other classes"
        elif norm_method in ["normalize all classes", "normalize other classes"]:
            self.normalization_method = norm_method
        else:
            self.initialization_errors.append(
                f"""Invalid normalization method selected.
                                         \tEntered value: {norm_method}
                                         \tValid entries:
                                         \t\t- normalize all classes
                                         \t\t- normalize other classes"""
            )

        self.dist_pdf = (
            None  # an array of the distributions PMF/PDF based on sim_bins_mid
        )
        self.dist_type = "discrete"  # specifies "discrete" vs "continuous"
        self.slider_keys = []  # keys to access slider values in st.session_state
        self.num_classes = 6  # number of classes
        self.sim_bin_width = 1  # width of the simulation bin #! currently unused.
        self.sim_bins_markers = None  # edge limits for each bin edge for creating historgrams and CDF calculations
        self.sim_bins_mid = None  # mid pont for plotting
        self.sim_bins_cnts = np.zeros(
            self.num_classes
        )  # binned data from the simulation
        self.sim_total_entries = 0  # total number of simulation entries
        self.dist_values = (
            np.ones(self.num_classes) / self.num_classes
        )  # list of the distribution properties
        self.plot_rng = np.array(
            [-1, 2]
        )  # list of two values indicating the required range for this distribution to be plotted over

        self.plot_dist_clr = None  # specifies the color of the distribution in the plot
        self.plot_sim_clr = None  # specifies the color of the simulation in the plot

        self.reset_sim()
        self._update_dist_pdf()

    def create_sliders(self):  # create the required class sliders
        """Creates the sliders that are required to define the distribution.
        """
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
        self, x: float
    ) -> float:  # cdf of the distribution over the range x_0 --> x_1
        """Returns the probability of obtaining a random value between 
            -infinity and x for the mathematical definition of the 
            distribution.

        :param float x: Upper limit for value range.
        :float: Probability for the provided range.
        """
        idxs = np.arange(self.num_classes) <= x
        cdf = np.sum(self.dist_values[idxs])

        return cdf

    def range_probability(
        self, x_0: Union[List, np.ndarray, tuple, float], x_1: Optional[float] = None
    ) -> float:  # cdf of the distribution over the range x_0 --> x_1
        """Returns a single float value for the probability of obtaining a 
            value between x_0 and x_1 for the ideal distribution, i.e. based 
            on the mathematical definition of the distribution.

        :param Union[List, np.ndarray, tuple, float] x_0: This can be either 
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

    def sim_cdf(self, x: float) -> float:  # cdf of the simulation results
        """Returns the probability of obtaining a random value between 
            -infinity and x for the simulation results.

        :param float x: Upper limit for value range.
        :float: Probability for the provided range.
        """
        # safety measure incase the simulation has not been run yet
        if self.sim_total_entries == 0:
            return 0

        bins = self.sim_bins_mid <= x

        # return sum of cdf range, normalized using total number of sim entries
        return np.sum(self.sim_bins_cnts[bins]) / self.sim_total_entries

    def sim_range_probability(
        self, x_0: Union[List, np.ndarray, tuple, float], x_1: Optional[float] = None
    ) -> float:  # cdf of the simulation results
        """Returns a single float value for the probability of obtaining a 
            value between x_0 and x_1 for the simulation results.

        :param Union[List, np.ndarray, tuple, float] x_0: This can be either 
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
        self.sim_bins_mid = np.arange(self.num_classes)
        self.sim_bins_cnts = np.zeros(self.num_classes)
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
        figure.add_bar(
            x=self.sim_bins_mid,
            y=self.dist_pdf,
            name="Bernoulli Expectation",
            marker_color=self.plot_dist_clr,
        )

        if self.sim_total_entries > 0:
            sim_bin_cnts = self._get_sim_bins_normalized()
            figure.add_bar(
                x=self.sim_bins_mid,
                y=sim_bin_cnts,
                marker_color=self.plot_sim_clr,
                name="Bernoulli Simulation",
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

    #! set this to take a list
    def _create_dist(  #! add documentation
        self, class_probabilities: Optional[List[float]] = None,
    ):  # creates the scipy.stats distribution
        """
        #! add documentation
        """
        if class_probabilities is not None:
            self.dist_values = np.array(class_probabilities)
            self.num_classes = len(class_probabilities)

    def _update_sliders(
        self,
    ):  # updates the slider values to the current class ratio values
        """Updates the slider values to match the class values. This is often
            done after the class ratios have been normalized.
        """
        for i, key in enumerate(self.slider_keys):
            self.session_state[key] = self.dist_values[i]

    def _update_class_ratios(self):  # get the values for each slider
        """Updates the class variable self.dist_values to reflect the current
            slider values.
        """
        self.reset_sim()
        for i, key in enumerate(self.slider_keys):
            self.dist_values[i] = self.session_state[key]

    def _normalize_class_ratios(self, cls_updated: int):
        """Function used to normalize the class values.

        :param int cls_updated: Index of the class that was updated.
        """
        self._update_class_ratios()
        updated_val = self.session_state[self.key_root + "_" + str(cls_updated)]
        self.dist_values[cls_updated] = updated_val

        if self.normalization_method == "normalize all classes":
            # normalize all of the classes together
            self._normalize_all_classes()
        elif self.normalization_method == "normalize other classes":
            # normalize all other classes to fit the class being adjusted
            self._normalize_other_classes(cls_updated)

        # update slider values
        self._update_sliders()

    def _normalize_all_classes(self):  # normalize all class entries
        """This function normalizes all elements of an array such that they sum 
            up to 1. All classes are treated equally and updated based on their 
            current ratios
        """
        # sum elements
        sum_total = np.sum(self.dist_values)

        for idx in np.arange(self.num_classes):
            # normalize each individual element
            self.dist_values[idx] = self.dist_values[idx] / sum_total

    def _normalize_other_classes(
        self, cls_updated: int
    ):  # normalize all entries except the value modified
        """This function normalizes all minus 1 elements of an array. This is used 
            so a single class entry can have a specified value and all other class
            entries will be normalized to ensure the entire array is normalized.

        :param int cls_updated: Index of the class to normalize based upon.
        """
        # create a list of the elements of the array that are to be updated
        idxs = list(np.arange(self.num_classes))
        del idxs[cls_updated]

        # obtain the value of the class we are updating based upon
        updated_val = self.dist_values[cls_updated]

        # if all other elements are 0, set them to 1 to ensure scaling works correctly
        if sum(self.dist_values[idxs]) == 0:
            self.dist_values[idxs] = np.ones(idxs)

        # obtain current sum_ratio
        sum_ratio = (1 - updated_val) / sum(self.dist_values[idxs])

        for idx in idxs:
            # update each individual element except for the class entry that was updated externally
            self.dist_values[idx] = sum_ratio * self.dist_values[idx]

    def _get_sim_bins_normalized(
        self,
    ) -> Optional[
        List
    ]:  # returns the current sim_bins counts normalized to match the distribution curve
        """If a simulation has been run, returns the number of entries in each 
            bin, normalized by the total number of random values used in the 
            simulation.

        :Optional[List]: If a simulation has been run a list of float values 
            will be returned. If no simulation has been run, None is returned.
        """

        if self.sim_total_entries == 0:
            return None

        return self.sim_bins_cnts / self.sim_total_entries

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
        self.dist_pdf = self.dist_values

    def _update_plot_rng(
        self,
    ):  # updates the required plot range based on current distribution parameters
        self.plot_rng = np.array([-1, self.num_classes])

