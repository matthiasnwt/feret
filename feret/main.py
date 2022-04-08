import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time, scipy.optimize
from scipy.spatial.distance import pdist

class Parameters():
    def __init__(
        self,
        maxferet,
        minferet):

        self.maxferet = maxferet
        self.minferet = minferet


class Calculater():

    def __init__(self, img, **kwargs):

        self.img = img

        if 'precision' in kwargs:
            self.precesion = kwargs['precision']
        else:
            self.precesion = 1

        if 'edge' in kwargs:
            self.edge = kwargs['edge']
        else:
            self.edge = False


        self.degs = np.deg2rad(np.arange(0, 180+self.precesion, self.precesion))
        self.ferets = np.zeros(len(self.degs))

        self.find_points()

        (self.y0, self.x0) = ndimage.center_of_mass(self.contour)


        self.calculate_ferets()


        self.minimize_feret()

        self.calculate_maxferet()


    def calculate_maxferet(self):
        """
        The maxferet is defined as the maximum euclidean
        distance between two points. pdist calculates
        all the distances between the points and than
        the maximum is taken from all those.

        """

        self.maxferet = max(pdist(self.points.T, "euclidean"))
    

    def find_points(self):
        """
        Method find the points which are used to calcualte feret diameter

        """

        if self.edge:
            ys, xs = np.nonzero(self.img)
            new_xs = np.concatenate((xs + 0.5, xs + 0.5, xs - 0.5, xs - 0.5, xs, xs + 0.5, xs - 0.5, xs, xs))
            new_ys = np.concatenate((ys + 0.5, ys - 0.5, ys + 0.5, ys - 0.5, ys, ys, ys, ys + 0.5, ys - 0.5))
            
            new_ys, new_xs = (new_ys * 2).astype(int), (new_xs * 2).astype(int)
            new_points = np.array([new_ys, new_xs])

            self.contour = np.zeros((self.img.shape[0] * 2, self.img.shape[1] * 2))
            self.contour[tuple(new_points)] = 1

            edm = ndimage.distance_transform_edt(self.contour)
            self.contour[edm > 1] = 0
            self.points = np.array(np.nonzero(self.contour))
        else:
            self.contour = np.copy(self.img)
            edm = ndimage.distance_transform_edt(self.contour)
            self.contour[edm > 1] = 0
            self.points = np.array(np.nonzero(self.contour))



    def calculate_center(self):
        """
        Method caluclates the center of the binary image.

        """
        



    def calculate_distances(self, angle):
        """ 
        Method calculates the distance of two points at a givin angle.

        """
        m = np.tan(angle)
        ds = np.cos(angle)*(self.y0-self.points[0])-np.sin(angle)*(self.x0-self.points[1])
        max_i = np.argmax(ds)
        min_i = np.argmin(ds)

        t_max = self.points.T[max_i][0] - m * self.points.T[max_i][1]
        t_min = self.points.T[min_i][0] - m * self.points.T[min_i][1]

        feret = np.abs(t_max - t_min) / np.sqrt(1+m**2)

        return feret


    def minimize_feret(self):
        """


        """

        res_minferet = scipy.optimize.minimize(
            self.calculate_distances, 
            x0=self.minferet_angle,  
            bounds=((0., np.pi),))

        self.minferet = res_minferet.fun
       

    def calculate_ferets(self):
        """


        """

        
        for i, angle in enumerate(self.degs):
            feret = self.calculate_distances(angle)

            self.ferets[i] = feret



        self.maxferet_initial = np.max(self.ferets)
        self.minferet_initial = np.min(self.ferets)

        self.minferet_index = np.where(self.ferets == self.minferet_initial)
        self.maxferet_index = np.where(self.ferets == self.maxferet_initial)

        self.minferet_angle = self.degs[self.minferet_index]
        self.maxferet_angle = self.degs[self.maxferet_index]

        





    def __call__(self):
        if self.edge:
            return self.maxferet / 2, self.minferet / 2
        else:
            return self.maxferet, self.minferet



if __name__ == '__main__':
    import tifffile as tif
    import time

    img = tif.imread('24_binary.tif')

    t0 = time.perf_counter()
    maxferet, minferet = Calculater(img)()
    results = Parameters(maxferet, minferet)

    t1 = time.perf_counter()

    print(t1 - t0, 'sekunden')
    print(results.maxferet, results.minferet)

