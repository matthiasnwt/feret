import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time, scipy.optimize
from scipy.spatial.distance import pdist, squareform



class Calculater():

    def __init__(self, img, edge):

        self.img = img
        self.edge = edge

        self.find_points()
        self.y0, self.x0 = ndimage.center_of_mass(self.contour)


    def find_points(self):
        """
        Method find the points which are used to calcualte feret diameter

        """

        if self.edge:
            ys, xs = np.nonzero(self.img)
            new_xs = np.concatenate(
                (xs+0.5, xs+0.5, xs-0.5, xs-0.5, xs, xs+0.5, xs-0.5, xs, xs))
            new_ys = np.concatenate(
                (ys+0.5, ys-0.5, ys+0.5, ys-0.5, ys, ys, ys, ys+0.5, ys-0.5))
            
            new_ys, new_xs = (new_ys*2).astype(int), (new_xs*2).astype(int)
            new_points = np.array([new_ys, new_xs])

            self.contour = np.zeros(
                (self.img.shape[0] * 2, self.img.shape[1] * 2))
            self.contour[tuple(new_points)] = 1
        else:
            self.contour = np.copy(self.img)

        edm = ndimage.distance_transform_edt(self.contour)
        self.contour[edm > 1] = 0
        self.points = np.array(np.nonzero(self.contour))


    def calculate_maxferet(self):
        """
        The maxferet is defined as the maximum euclidean
        distance between two points. pdist calculates
        all the distances between the points and than
        the maximum is taken from all those.

        """

        pdists = pdist(self.points.T, "euclidean")

        self.maxf = np.max(pdists)

        maxf_coords_index = np.where(squareform(pdists) == self.maxf)[0]
        self.maxf_coords = self.points.T[maxf_coords_index]

        if self.edge:
            self.maxf /= 2


    def calculate_minferet(self):
        """
        To calcualte the minferet, first the distances for all
        the angles from 0-180° are calculted and the minimum
        ist used as an initial guess for a function minimization.

        """

        self.get_initial_minf_estimation()
        self.minimize_feret()

        if self.edge:
            self.minf /= 2
    

    def minimize_feret(self):
        """
        The approximated minferet is calcualted using
        a minimazation with the angle.

        """

        res_minferet = scipy.optimize.minimize(
            self.calculate_distances, 
            x0=self.minferet_angle,  
            bounds=((0., np.pi),))

        self.minf = res_minferet.fun
       

    def get_initial_minf_estimation(self):
        """
        This method finds the initial guess for the minimum feret
        diameter.


        """
        degs = np.deg2rad(np.arange(0, 180.1, 0.1))
        distances = np.empty((len(degs)))
        
        for i, angle in enumerate(degs):
            distances[i] = self.calculate_distances(angle)


        self.minf_initial = np.min(distances)
        self.minf_initial_index = np.where(distances == self.minf_initial)
        self.minf_angle = degs[self.minf_initial_index]


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


    def __call__(self):
        return self.maxf, self.minf


