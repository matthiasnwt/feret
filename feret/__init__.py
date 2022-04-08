from feret.FeretDiameter import FeretDiameter

def calculate(img, **kwargs):

    feret = FeretDiameter(img, **kwargs)
    # feret.plot()
    return feret()