import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
import cv2 as cv

class Calculater():
    def __init__(self, img, edge):
        self.img = img
        self.edge = edge

        self.find_convexhull()
        self.y0, self.x0 = ndimage.center_of_mass(self.hull)

    def find_convexhull(self):
        # Method calculates convexhull.
        # If edge flag is set, it uses the corners not centers
        if self.edge:
            ys, xs = np.nonzero(self.img)
            new_xs = np.concatenate(
                (xs+0.5, xs+0.5, xs-0.5, xs-0.5, xs, xs+0.5, xs-0.5, xs, xs))
            new_ys = np.concatenate(
                (ys+0.5, ys-0.5, ys+0.5, ys-0.5, ys, ys, ys, ys+0.5, ys-0.5))
            new_ys, new_xs = (new_ys*2).astype(int), (new_xs*2).astype(int)
            self.hull = cv.convexHull(np.array([new_ys, new_xs]).T).T.reshape(2, -1)
        else:
            self.hull = cv.convexHull(np.transpose(np.nonzero(self.img))).T.reshape(2, -1)


    def calculate_minferet(self):
        """
        Method calculates the exact minimum feret diameter.
        The result is equal to imagejs minferet.

        """
        length = len(self.hull.T)

        Ds = np.empty(length)
        ps = np.empty((length, 3, 2))


        for i in range(length):
            p1 = self.hull.T[i]
            p2 = self.hull.T[(i+1) % length]
            
            ds = np.abs(np.cross(p2-p1, p1-self.hull.T)/norm(p2-p1))

            d_i = np.where(ds == Ds[i])[0][0]
            p3 = self.hull.T[d_i]

            Ds[i] = np.max(ds)
            ps[i] = np.array((p1, p2, p3))

        self.minf = np.min(Ds)

        minf_index = np.where(Ds == self.minf)[0][0]

        (y0, x0), (y1, x1), (y2, x2) = ps[minf_index]

        m = (y0 - y1) / (x0 - x1)
        t = y0 - m * x0

        self.minf_angle = np.arctan(m)

        if self.edge:
            self.minf /= 2


    def calculate_maxferet(self):
        """
        The maxferet is defined as the maximum euclidean
        distance between two points. pdist calculates
        all the distances between the points and than
        the maximum is taken from all those.

        """
        pdists = pdist(self.hull.T, "euclidean")

        self.maxf = np.max(pdists)

        maxf_coords_index = np.where(squareform(pdists) == self.maxf)[0]
        self.maxf_coords = self.hull.T[maxf_coords_index]

        ((y0, x0), (y1, x1)) = self.maxf_coords

        m = (y0 - y1) / (x0 - x1)
        t = y0 - m * x0

        self.maxf_angle = np.arctan(m) + np.pi/2

        if self.edge:
            self.maxf /= 2


    def calculate_maxferet90(self):
        """
        Method calculates the feret diameter which
        is 90 degree to the maximum feret diameter.
        It first checks if the angle of the maximum
        feret diameter is already calculatet. If not
        it calls the maxferet function.

        """
        self.maxf90_angle =self.maxf_angle + np.pi/2
        self.maxf90 = self.calculate_distances(self.maxf90_angle)
        
        if self.edge:
            self.maxf90 /= 2


    def calculate_minferet90(self):
        """
        Method calculates the feret diameter which
        is 90 degree to the minimum feret diameter.
        It first checks if the angle of the minimum
        feret diameter is already calculatet. If not
        it calls the minferet function.

        """        
        self.minf90_angle =self.minf_angle + np.pi/2
        self.minf90 = self.calculate_distances(self.minf90_angle)

        if self.edge:
            self.minf90 /= 2


    def calculate_distances(self, a):
        """ 
        Method calculates the distance of two points at a givin angle.

        Args:
            a (float): angle (in rad)

        Returns:
            distance (float): caliper distance for angle a
        """
        m = np.tan(a)
        ds = np.cos(a)*(self.y0-self.hull[0])-np.sin(a)*(self.x0-self.hull[1])
        max_i = np.argmax(ds)
        min_i = np.argmin(ds)

        t_max = self.hull.T[max_i][0] - m * self.hull.T[max_i][1]
        t_min = self.hull.T[min_i][0] - m * self.hull.T[min_i][1]

        distance = np.abs(t_max - t_min) / np.sqrt(1+m**2)

        return distance