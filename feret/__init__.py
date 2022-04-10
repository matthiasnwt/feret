from feret.main import Calculater

def calc(img, edge=False):
    """
    Calculate the true maximum feret diameter (minferet)
    and the approximated minimum feret diameter (minferet).

    result = calc(img)
    maxferet, minferet = result.maxferet, result.minferet

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
    Calculate the true maximum feret diameter (minferet)
    and the approximated minimum feret diameter (minferet).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        maxferet (float): maximum feret diameter
        minferet (float): minimum feret diameter
    """

    feret_calc = calc(img, edge)
    return feret_calc.maxferet, feret_calc.minferet


def max(img, edge=False):
    """
    Calculate the true maximum feret diameter (minferet).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        maxferet (float): maximum feret diameter
    """

    feret_calc = Calculater(img, edge)
    feret_calc.calculate_maxferet()
    return feret_calc.maxferet


def min(img, edge=False):
    """
    Calculate the approximated minimum feret diameter (minferet).

    Args:
        img (numpy.ndarray): binary-image
        edge (boolean): use edges (vertices) or centers

    Returns:
        minferet (float): minimum feret diameter
    """

    feret_calc = Calculater(img, edge)
    feret_calc.calculate_minferet()
    return feret_calc.minferet