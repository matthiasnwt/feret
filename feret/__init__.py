from main import Calculater
import numpy as np

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
    feret_calc.calculate_maxferet()
    feret_calc.calculate_minferet()
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
    return feret_calc.maxf, feret_calc.minf


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


if __name__ == '__main__':
    img = np.load('img.npy')
    maxf, minf = all(img, edge=True)
    print(maxf, minf)
