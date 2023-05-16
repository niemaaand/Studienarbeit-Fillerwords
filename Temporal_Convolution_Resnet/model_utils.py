import os


def read_lines(datapath):
    # Read all the lines in a text file and return as a list
    with open(datapath, 'r') as f:
        lines = f.read().splitlines()
    return lines

