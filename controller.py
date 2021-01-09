from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure

import utils


class Controller:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
