#!/usr/bin/env python

"""
Small utility to help test for normality assumptions.
It is a stand-alone script working on delimited text files.

This is a graphical utility: It will generate the QQ plot for the normal
distribution.

"""

from __future__ import print_function

import logging
import argparse
import sys
import gzip

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sbn
except ImportError:
    pass  # Seaborn is not required.


def main(filename, field, delimiter):

    data = extract_column(filename, field, delimiter)


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
