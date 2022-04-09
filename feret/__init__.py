from feret.main import Calculater

def calc(img, edge=False):
    feret_calc = Calculater(img, edge)
    feret_calc.calculate_maxferet()
    feret_calc.calculate_minferet()
    return feret_calc

def all(img, edge=False):
    feret_calc = calc(img, edge)
    return feret_calc.maxferet, feret_calc.minferet

def max(img, edge=False):
    feret_calc = Calculater(img, edge)
    feret_calc.calculate_maxferet()
    return feret_calc.maxferet

def min(img, edge=False):
    feret_calc = Calculater(img, edge)
    feret_calc.calculate_minferet()
    return feret_calc.minferet