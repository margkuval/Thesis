"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np
import seaborn as sns


class Logger(object):
    """
    Helper class which stores a group of numpy arrays that store the information
    of the evolutionary algorithm over each iteration. We can define the parameters
    to log by passing a dictionary to the Logger class. It also plots the stored values
    as a graph.
    """

    def __init__(self, iter_log):
        """
        Initializes the Logger object creating the dictionary of arrays
        to store the logged values
        :param logs:
        """
        self.values = {}
        self.log_size = 0
        self.iter_log = iter_log

    def add_log(self, logs):
        """
        Adds a new item to log into the [values] dictionary
        :param logs:
        """

        # Validate that we do not add logs after the logging process
        # has started
        for key in self.values:
            assert (len(self.values[key]) == 0)

        # Add as many keys as given in logs
        for key in logs:
            assert (self.logs[key] == None)
            self.values[key] = np.array([])

    def log(self, values, count_it=True):
        """
        Stores the logged values into the [values] dictionary
        :param values:
        """
        for key in values:
            # Check if it is an array, in this case store it vertically (vstack)
            if str(type(values[key])) == "<type 'numpy.ndarray'>":
                self.values[key] = np.vstack((self.values[key], values[key])) if key in self.values else  np.array(
                    values[key])
            else:
                # If the element in value[key] is a number we transform it to an array
                if str(type(values[key])) == "<type 'numpy.float64'>":
                    values[key] = np.array([values[key]])

                # add the value to the log of values
                self.values[key] = np.hstack((self.values[key], values[key])) if key in self.values else  np.array(
                    values[key])

        # Print the iteration result
        if self.iter_log > 0 and count_it:
            if self.iter_log and (self.log_size + 1) % self.iter_log == 0:
                self.print_log(self.log_size)

        self.log_size += 1 if count_it else 0

    def get_log(self, key):
        """
        :param key:
        :return: return one of the logged values
        """
        return self.values[key]

    def print_description(self, problem, elements_print=None, offset=30):
        """
        :param problem:
        :param elements_print:
        :return:
        """
        print ("-----------------------------------------")
        for elem in problem:
            print ('{:' + str(offset) + '}').format(elem) + '| ' + str(problem[elem])
        print ("-----------------------------------------")
        if elements_print:
            for elem in elements_print:
                print ('{:' + str(offset) + '}').format(elem) + '| ' + str(elements_print[elem])
            print ("-----------------------------------------\n")

    def print_log(self, iteration):
        """
        print the result at the iteration [iteration] of the logged
                values. Useful to keep track of the process
        :param iteration:
        """
        res = "iteration " + str(iteration + 1) + " \n\t"
        for key in self.values:
            # Avoid printing values logged as matrix
            res += " " + key + " " + str(self.values[key][iteration]) + " \n\t"
        print(res)

    def plot(self, keys_plot, problem, show=True):
        """
        Draws a plot with the logged values
        :param logs:
        """

        # create a palette of colours with the number of keys
        keys = [key for key in self.values]
        palette = sns.color_palette("hls", len(keys))

        i = 0
        plotted_keys = np.array([])
        for key in keys_plot:
            # Avoid printing values logged as matrix
            if len(self.values[key].shape) == 1:
                sns.plt.plot(np.arange(0, self.log_size), np.abs(self.values[key]), color=palette[i])
                i += 1
                plotted_keys = np.append(plotted_keys, key)

        # Create the legend
        sns.plt.legend(plotted_keys, loc='upper right')
        sns.plt.xlabel("Iterations")
        sns.plt.ylabel("Fitness")
        sns.plt.title("RUN " + problem)
        sns.plt.savefig('results/RUN' + problem + '.pdf')
        if show:
            sns.plt.show()
        sns.plt.clf()