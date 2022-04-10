import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import time, scipy.optimize
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import norm
import cv2 as cv

class Calculater():
    def __init__(self, img, edge):
        self.img = img
        self.edge = edge

        self.find_points()
        self.y0, self.x0 = ndimage.center_of_mass(self.contour)


    def calculate_minferet_analytical(self):

        self.minf = -1

        length = len(self.hull.T)

   

        ds = []

        for i in range(length):
            p1 = self.hull.T[i]
            p2 = self.hull.T[(i+1) % length]
            for j in range(length):
                p3 = self.hull.T[j]

                if p1[0] == p3[0] and p1[1] == p3[1]:
                    continue
                if p2[0] == p3[0] and p2[1] == p3[1]:
                    continue


                v1 = p2 - p1
                v2 = p3 - p1
                v3 = p1 - p2
                v4 = p3 - p2

                dot1 = np.dot(v1, v2)
                dot2 = np.dot(v3, v4)

                if dot1 <= 0 or dot2 <= 0:
                    continue

                d = np.abs(np.cross(p2-p1, p1-p3)/norm(p2-p1))
                ds.append(d)

                m = (p2[0] - p1[0]) / (p2[1] - p1[1])
                t12 = p1[0] - m * p1[1]
                t3 = p3[0] - m * p3[1]


                xs = np.linspace(0, self.img.shape[1])
                ys12 = m * xs + t12
                ys3 = m * xs + t3

                fig, ax = plt.subplots()
                ax.scatter(self.hull[1], self.hull[0], color='green', marker='o')
                ax.scatter(p1[1], p1[0], marker='x', color='red')
                ax.scatter(p2[1], p2[0], marker='x', color='red')
                ax.scatter(p3[1], p3[0], marker='x', color='red')
                ax.text(p1[1], p1[0], 'p1')
                ax.text(p2[1], p2[0], 'p2')
                ax.text(p3[1], p3[0], 'p3')

                ax.plot(xs, ys12)
                ax.plot(xs, ys3)

                    

                ax.set_title(f'{d:.5f} {int(dot1)} {int(dot2)} {p1} {p2} {p3}')
                ax.set_aspect('equal', adjustable='box')
                plt.ylim((0, self.img.shape[0]))
                plt.show()
                

                
        print(len(ds))

        plt.scatter(np.arange(len(ds)), ds)
        plt.show()

        self.minf = np.min(ds)


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

        # Find contour of img.
        edm = ndimage.distance_transform_edt(self.contour)
        self.contour[edm > 1] = 0
        self.points = np.array(np.nonzero(self.contour))
        self.hull = cv.convexHull(self.points.T).T.reshape(2, -1)
        

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
        self.maxf_coords = self.points.T[maxf_coords_index]

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
        

    def minimize_feret(self):
        """
        The approximated minferet is calcualted using
        a minimazation with the angle.

        """

        res_minferet = scipy.optimize.minimize(
            self.calculate_distances, 
            x0=self.minf_angle,  
            bounds=((0., np.pi),))

        self.minf = res_minferet.fun
        self.minf_angle = res_minferet.x[0]

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


    def calculate_distances(self, a):
        """ 
        Method calculates the distance of two points at a givin angle.

        Args:
            a (float): angle (in rad)

        Returns:
            distance (float): caliper distance for angle a
        """
        m = np.tan(a)
        ds = np.cos(a)*(self.y0-self.points[0])-np.sin(a)*(self.x0-self.points[1])
        max_i = np.argmax(ds)
        min_i = np.argmin(ds)

        t_max = self.points.T[max_i][0] - m * self.points.T[max_i][1]
        t_min = self.points.T[min_i][0] - m * self.points.T[min_i][1]

        distance = np.abs(t_max - t_min) / np.sqrt(1+m**2)

        return distance


    def __call__(self):
        return self.maxf, self.minf


