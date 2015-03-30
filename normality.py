#!/usr/bin/env python

"""
Small utility to help test for normality assumptions.
It is a stand-alone script working on delimited text files.

This is a graphical utility: It will generate the QQ plot for the normal
distribution.

"""

from __future__ import print_function, division

import logging
import argparse
import sys
import gzip

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

try:
    import seaborn as sbn
except ImportError:
    pass  # Seaborn is not required.


def main(filename, field, delimiter):

    data = extract_column(filename, field, delimiter)
    data = data[~np.isnan(data)]

    fig, axes = plt.subplots(2, 1)

    create_histogram(data, axes[0])

    create_qq_plot(data, axes[1])

    plt.show()


def create_histogram(data, ax):
    """Create an histogram with a fitted normal curve."""

    mu = np.mean(data)
    sigma = np.std(data)

    # See http://stackoverflow.com/questions/9767241/setting-a-relative-frequency-in-a-matplotlib-histogram
    # for the relative frequency.
    n, bins, patches = ax.hist(data, bins=60, edgecolor="#EDF1F4",
                               facecolor="#112029",
                               weights=np.zeros_like(data) + 1. / data.size)

    # Overlap a normal curve.
    normal_densities = 1 / (sigma * np.sqrt(2 * np.pi))
    normal_densities *= np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2))
    ax.plot(bins, normal_densities, color="#99CC00",
            label="$\mathcal{{N}}({:.3f}, {:.3f}^2)$".format(mu, sigma))

    ax.set_ylabel("Relative frequency")
    ax.legend()


def create_qq_plot(data, ax):
    """Create the Normal QQ plot."""
    quantiles, fit = scipy.stats.probplot(data, dist="norm")

    # Order statistic medians, ordered responses.
    osm, osr = quantiles 

    slope, intercept, r = fit

    ax.scatter(osm, osr, color="black", marker="o", s=10)
    xs = np.arange(*ax.get_xlim())
    ax.plot(xs, slope * xs + intercept, "--", color="#6D784B",
            label="$R^2 = {:.4f}$".format(r))
    ax.legend(loc="lower right")

    ax.set_xlabel("Quantile")
    ax.set_ylabel("Ordered Values")


def extract_column(filename, field, delimiter):
    """ Extract a column of data from a file.
    
    :param filename: The filename of the file containing the data.
                     If None is passed, the script will read from stdin.
    :type filename: str

    :param field: The number of the column of interest (starting at one).
    :type field: int

    :param delimiter: The separator for the columns.
    :type delimiter: str

    """
    if field is None:
        check_single_column = True
    else:
        check_single_column = False
        field -= 1  # User should use 1 based indexing.
        if field < 0:
            raise Exception("Use 1 based indexing for the --field argument.")

    if filename.endswith(".gz"):
        opener = gzip.open
    elif filename == "-":
        # Dummy function to read from stdin by default.
        opener = lambda x: sys.stdin
    else:
        opener = open

    f = opener(filename)

    try:
        data = []
        for line in f:
            # Split the line.
            line = line.rstrip("\n").split(delimiter)

            if check_single_column:
                if len(line) != 1:
                    raise Exception("You need to provide a --field for files "
                                    "with more than one column.")
                value = line[0]
            # Check if we can index the required position.
            elif len(line) > field:
                value = line[field]
            else:
                value = None

            data.append(value)

    finally:
        f.close()

    # Try casting everything to float.
    for i, value in enumerate(data):
        try:
            data[i] = float(value)
        except ValueError:
            data[i] = None

    data = np.array(data, dtype=float)
    logging.info("Read {} values ({} missing) from file {}.".format(
        np.sum(~np.isnan(data)),
        np.sum(np.isnan(data)),
        filename if filename != "-" else "stdin"
    ))

    return data


def parse_args():
    """Parse the command line arguments. """

    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "file",
        help="The file containing the data (default: - (stdin)).",
        default="-",
        type=str,
    )

    parser.add_argument(
        "--field", "-f",
        help="The field number (starting from 1) of the variable of interest. "
             "This argument is not required if there is only one column.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--delimiter", "-d",
        help="The field delimiter in the text file (default: TAB).",
        default="\t"
    )

    parser.add_argument(
        "--debug",
        help="Enable the debugging messages (logging).",
        action="store_true",
    )

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    main(args.file, args.field, args.delimiter)

if __name__ == "__main__":
    args = parse_args()
