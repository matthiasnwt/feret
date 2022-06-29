#
from matplotlib import pyplot as plt

# from main import Calculater

from feret.main import Calculater


def calc(img, edge=False):
    """
    Calculate the true maximum feret diameter (minf)
    and the approximated minimum feret diameter (minf).

    result = calc(img)
    maxf, minf = result.maxf, result.minf

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        Calculater (object): calculator class.
    """
    feret_calc = Calculater(img, edge)
    feret_calc.calculate_minferet()
    feret_calc.calculate_minferet90()
    feret_calc.calculate_maxferet()
    feret_calc.calculate_maxferet90()
    return feret_calc


def all(img, edge=False):
    """
    Calculate the true maximum feret diameter (minf)
    and the approximated minimum feret diameter (minf).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        maxf (float): maximum feret diameter
        minf (float): minimum feret diameter
    """

    feret_calc = calc(img, edge)
    return feret_calc.maxf, feret_calc.minf, feret_calc.minf90, feret_calc.maxf90


def plot(img, edge=False):
    """
    Plot of the results.

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    """
    feret_calc = calc(img, edge)
    feret_calc.plot()


def max(img, edge=False):
    """
    Calculate the true maximum feret diameter (minf).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        maxf (float): maximum feret diameter
    """

    feret_calc = Calculater(img, edge)
    feret_calc.calculate_maxferet()
    return feret_calc.maxf


def min(img, edge=False):
    """
    Calculate the approximated minimum feret diameter (minf).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        minf (float): minimum feret diameter
    """

    feret_calc = Calculater(img, edge)
    feret_calc.calculate_minferet()
    return feret_calc.minf


def min90(img, edge=False):
    """
    Calculate the approximated feret diameter 90  
    degree (minf90) to minimum feret diameter (minf).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        minf90 (float): minimum feret diameter 90 degree
    """

    feret_calc = Calculater(img, edge)
    feret_calc.calculate_minferet()
    feret_calc.calculate_minferet90()
    return feret_calc.minf90


def max90(img, edge=False):
    """
    Calculate the approximated feret diameter 90  
    degree (maxf90) to maximum feret diameter (minf).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        maxf90 (float): maximum feret diameter 90 degree
    """

    feret_calc = Calculater(img, edge)
    feret_calc.calculate_maxferet()
    feret_calc.calculate_maxferet90()
    return feret_calc.maxf90

# if __name__ == '__main__':
#
#     import numpy as np
#     import time
#     import tifffile as tif
#     # img = np.load('img.npy')
#     img = tif.imread('10243_binary.tif').T
#     img = tif.imread('11300_binary_verrgroessert.tif')
#
#     plot(img, edge=True)
#
    # print(min90(img))
