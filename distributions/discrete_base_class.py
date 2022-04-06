from typing import Optional, Union, List
import streamlit as st
import scipy.stats as stats
import numpy as np
import plotly.graph_objects as go


class discrete_base_cls:
    def __init__(
        self,
        key_root: str,
        session_state,
        norm_method: Optional[str] = None,
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

        self.dist = None  # scipy.stats distribution
        self.dist_values = None  # values defining the distribution
        self.dist_pdf = (
            None  # an array of the distributions PMF/PDF based on sim_bins_mid
        )
        self.dist_pdf_max = 0  # max value of the distributions PMF/PDF
        self.dist_type = "discrete"  # specifies "discrete" vs "continuous"
        self.slider_keys = []  # keys to access slider values in st.session_state
        self.sim_bins_markers = None  # edge limits for each bin edge for creating historgrams and CDF calculations
        self.sim_bins_mid = None  # mid pont for plotting
        self.sim_bins_cnts = None  # binned data from the simulation
        self.sim_total_entries = 0  # total number of simulation entries

        self.cdf_arr = (
            None  # stores the cdf values based on x-values set in self.sim_bins_mid
        )

        self.plot_dmn = np.array(
            [0, 1]
        )  # specifies the domain required for the distribution to plot over
        self.plot_rng = np.array(
            [0, 1]
        )  # specifies the y-range required for the distribution to plot over

        self.plot_dist_clr = None  # specifies the color of the distribution in the plot
        self.plot_sim_clr = None  # specifies the color of the simulation in the plot
        self.plot_mean_clr = (
            None  # specifies the color for the line inicating the distribution mean
        )

        self.plot_show_mean = (
            False  # mean of distribution should be included on the plot
        )

        self.plot_show_cdf = False  # cdf should be included on the plot
        self.plot_cdf_y2 = True  # cdf should have its own y-axis
        self.plot_cdf_clr = None  # specifies the color of the CDF in the plot

        self.dist_stats = None  # [mean, variance, skew, kurtosis]

    #! update docs
    def cdf(
        self,
        x: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:  # cdf of the distribution over the range x_0 --> x_1
        """Returns the probability of obtaining a random value between
            -infinity and x for the mathematical definition of the
            distribution.

        :param float x: Upper limit for value range.
        :float: Probability for the provided range.
        """
        return self.dist.cdf(x)

    def range_probability(
        self,
        x_0: Union[List[float], np.ndarray, tuple, float],
        x_1: Optional[float] = None,
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
        x_0, x_1 = self._parse_prob_dmn(x_0, x_1)

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
        self,
        x_0: Union[List, np.ndarray, tuple, float],
        x_1: Optional[float] = None,
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
        x_0, x_1 = self._parse_prob_dmn(x_0, x_1)

        cdf_0 = self.sim_cdf(x_0)
        cdf_1 = self.sim_cdf(x_1)

        # return sum of cdf range, normalized using total number of sim entries
        return cdf_1 - cdf_0

    def get_plot_domain(
        self,
    ) -> np.ndarray:  # calculate the range required for the distribution to be plotted
        """Returns the range that is required to plot the distribution.

        :np.ndarray: An array of two values indicating the range of the plot.
        """
        self._update_plot_dmn()
        return self.plot_dmn

    def get_plot_range(
        self,
    ) -> np.ndarray:  # returns the max y value required to plot this data
        """Returns the domain of plots generated by this function. This will
            consider both the mathematical definition of the function itself as
            well as any simulation results.

        :np.ndarray: An array of two values indicating the domain of the plot.
        """
        y_maxs = [self.dist_pdf_max]
        if self.sim_total_entries > 0:
            y_maxs.extend(self._get_sim_bins_normalized())

        # if self.session_state[self.key_root + "_plot-CDF"]:
        if self.plot_show_cdf and not self.plot_cdf_y2:
            y_maxs.append(1)

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
            name=self.key_root + " Expectation",
            marker_color=self.plot_dist_clr,
            showlegend=True,
            secondary_y=False,
        )

        if self.sim_total_entries > 0:
            sim_bin_cnts = self._get_sim_bins_normalized()
            figure.add_bar(
                x=self.sim_bins_mid,
                y=sim_bin_cnts,
                marker_color=self.plot_sim_clr,
                name=self.key_root + " Simulation",
                showlegend=True,
                secondary_y=False,
            )

        # if self.session_state[self.key_root + "_plot-mean"]:
        if self.plot_show_mean:
            x_mean = self.dist_stats[0]
            y_max = self.get_plot_range()[1]
            figure.add_trace(
                go.Scatter(
                    x=[x_mean, x_mean],
                    y=[0, y_max * 1.05],
                    line=dict(color=self.plot_mean_clr),
                    name=f"{self.key_root} mean: {x_mean:.3f}",
                ),
                secondary_y=False,
            )

        # if self.session_state[self.key_root + "_plot-CDF"]:
        if self.plot_show_cdf:
            self.plot_cdf(figure)
        else:
            figure.update_layout(yaxis2_showticklabels=False)

        return figure

    def update_sim_plot_data(self, figure: go.Figure, data_idx: int):
        """Updates the simulation data of the plotted figure. Requires the
            index of the data to be updated.

        :param go.Figure figure: A plotly.graph_object.Figure that is to be
            updated.
        :param int data_idx: The index of the data to be updated.
        :go.Figure: The updated Figure.
        """
        sim_vals = self._get_sim_bins_normalized()
        figure["data"][data_idx]["y"] = sim_vals

    def plot_cdf(self, figure: go.Figure):
        """Adds the CDF of a distribution to the provided figure.

        Args:
            figure (go.Figure): Figure object that the CDF needs to be added to.
        """
        if self.cdf_arr is None:
            self._update_dist_cdf()

        x_pnts = np.arange(self.plot_dmn[0], self.plot_dmn[1] + 1)

        x_0 = x_pnts[0]
        x_1 = x_pnts[1]
        cdf_val = 0

        use_markers = True
        x_0_marker = None
        x_1_marker = False
        if len(x_pnts) >= (15 + 4):
            use_markers = False
            x_0_marker = None
            x_1_marker = None

        for i in np.arange(len(x_pnts) - 2):
            if use_markers and i != 0:
                x_0_marker = True
            self._plot_hline(
                figure, x_0, x_1, cdf_val, self.plot_cdf_clr, x_0_marker, x_1_marker
            )

            x_0 = x_pnts[i + 1]
            x_1 = x_pnts[i + 2]
            cdf_val = self.cdf_arr[i]

        x_0_marker = True if use_markers else None
        self._plot_hline(figure, x_0, x_1, cdf_val, self.plot_cdf_clr, x_0_marker, None)

        if self.plot_cdf_y2:
            figure.update_layout(yaxis2_title="CDF Probability", yaxis2_range=[0, 1.05])
        else:
            figure.update_layout(yaxis2_showticklabels=False)

    #! Internal Functions

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
        for i, key in enumerate(self.slider_keys):
            self.dist_values[i] = self.session_state[key]

        self.reset_sim()
        self._create_dist()

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

        for idx in np.arange(len(self.dist_values)):
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
        idxs = list(np.arange(len(self.dist_values)))
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

    def _update_dist_pdf(
        self,
    ):  # returns the PMF/PDF of the distribution
        """Returns the PMF/PDF values of the distribution calculated at the
        middle of each bin used for the simulation results.
        """
        self._create_dist()
        self.dist_pdf = self.dist.pmf(self.sim_bins_mid)
        self.dist_pdf_max = np.max(self.dist_pdf)
        self._update_dist_cdf()

    def _update_dist_cdf(self):
        """Creates and stores an array of the cdf values across the range of
            values as specified by self.sim_bins_mid.

        Note: These values can be used to plot the cdf of a continuous
            distribution directly using self.sim_bins_mid for the x-axis values.
            Discrete distributions require the plot to include steps. In both
            cases, please use self.plot_cdf() to avoid potential issues.

        Uses class variables:
            - self.sim_bins_mid -> used to determine where cdf values should be
                calculated for.

        Sets results in class variables:
            - self.cdf_arr -> stores the calculated cdf values based on
                self.sim_bins_mid.
        """

        self.cdf_arr = self.cdf(self.sim_bins_mid)

    def _calc_dist_stats(self):
        dist_stats = self.dist.stats("mvsk")

        self.dist_stats = [float(entry) for entry in dist_stats]

    def _plot_hline(
        self,
        figure: go.Figure,
        x_0: float,
        x_1: float,
        y: float,
        color: str = "black",
        x_0_inclusive: Optional[bool] = None,
        x_1_inclusive: Optional[bool] = None,
    ):
        """Used to plot a single line. This function can be used recursively to
            create step wise functions like CDFs of discrete distributions.

        Args:
            figure (go.Figure): Figure to add line to.
            x_0 (float): x-value for the start of the line.
            x_1 (float): x-value for the end of the line.
            y (float): y-value for the line
            color (str, optional): Color of the line that will be plotted.
                Defaults to "black".
            x_0_inclusive (Optional[bool], optional): Sets the type of marker to
                be used at the start of the line. If None, no marker will be
                used. If True, a solid circle the same color as the line will be
                used. If False, a circle the same color as the line but with a
                white core will be used. Used to indicate if this value included
                in the lines domain. Defaults to None.
            x_1_inclusive (Optional[bool], optional): Sets the type of marker to
                be used at the end of the line. If None, no marker will be
                used. If True, a solid circle the same color as the line will be
                used. If False, a circle the same color as the line but with a
                white core will be used. Used to indicate if this value included
                in the lines domain. Defaults to None.
        """
        if color is None:
            color = "black"

        figure.add_trace(
            go.Scatter(
                x=[x_0, x_1],
                y=[y, y],
                line=dict(color=color),
                marker=dict(opacity=0),
                showlegend=False,
            ),
            secondary_y=self.plot_cdf_y2,
        )

        if x_0_inclusive is not None:
            figure.add_trace(
                go.Scatter(
                    x=[x_0, x_0],
                    y=[y, y],
                    line=dict(color=color),
                    marker=dict(size=12),
                    showlegend=False,
                ),
                secondary_y=self.plot_cdf_y2,
            )
            if not x_0_inclusive:
                figure.add_trace(
                    go.Scatter(
                        x=[x_0, x_0],
                        y=[y, y],
                        line=dict(color="white"),
                        marker=dict(size=6),
                        showlegend=False,
                    ),
                    secondary_y=self.plot_cdf_y2,
                )

        if x_1_inclusive is not None:
            figure.add_trace(
                go.Scatter(
                    x=[x_1, x_1],
                    y=[y, y],
                    line=dict(color=color),
                    marker=dict(size=12),
                    showlegend=False,
                ),
                secondary_y=self.plot_cdf_y2,
            )
            if not x_1_inclusive:
                figure.add_trace(
                    go.Scatter(
                        x=[x_1, x_1],
                        y=[y, y],
                        line=dict(color="white"),
                        marker=dict(size=6),
                        showlegend=False,
                    ),
                    secondary_y=self.plot_cdf_y2,
                )

    def _parse_prob_dmn(
        self,
        x_0: Union[List, np.ndarray, tuple, float],
        x_1: Optional[float] = None,
    ) -> np.ndarray:
        if isinstance(x_0, (list, np.ndarray, tuple)):
            if len(x_0) == 1:
                x_1 = x_0[0]
                x_0 = -np.inf
            else:
                x_1 = x_0[1]
                x_0 = x_0[0]
        elif x_1 is None:
            x_1 = x_0
            x_0 = -np.inf

        return np.array([x_0, x_1])

    #! Functions requiring a definition from the child class

    def _create_dist(self):
        pass

    def _update_plot_dmn(self):
        pass
