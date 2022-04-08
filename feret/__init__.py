from feret.main import Parameters, Calculater

def calc(img, **kwargs):

    maxferet, minferet = Calculater(img, **kwargs)()
    results = Parameters(maxferet, minferet)
    # feret.plot()
    return feretS