import matplotlib.pyplot as plt
import numpy as np

import random

class MetricGrid:
    def __init__(self, startPosRankCap, endPosRankCap, vals=None):
        self.__xLimit = endPosRankCap
        self.__yLimit = startPosRankCap

        if vals==None:
            self.__grid = []
            for x in range(0, self.__xLimit):
                self.__grid.append([])
                for y in range(0, self.__yLimit):
                    self.__grid[x].append(1.0/(x*y + 1))
        else:
            self.__grid = vals

    def setVal(self, x, y, val):
        self.__grid[x][y] = val

    def getVal(self, x, y):
        return self.__grid[x][y]

    def __str__(self):
        res = ""
        for x in range(0, self.__xLimit):
            for y in range(0, self.__yLimit-1):
                res += self.__grid[x][y] + ", "
            res += self.__grid[x][-1] + "\n"
        return res

    def showGridPlot(self):
        image = np.zeros((self.__xLimit, self.__yLimit))

        for x in range(0, self.__xLimit):
            for y in range(0, self.__yLimit):
                image[x][y] = self.__grid[x][y]

        row_labels = range(self.__xLimit)
        col_labels = range(self.__yLimit)
        plt.matshow(image, cmap="gray")
        plt.xticks(range(self.__yLimit), col_labels)
        plt.yticks(range(self.__xLimit), row_labels)
        plt.show()

    def showGridPlotValues(self):
        image = np.zeros((self.__xLimit, self.__yLimit))
        all_vals = []
        for x in range(0, self.__xLimit):
            for y in range(0, self.__yLimit):
                image[x][y] = self.__grid[x][y]
                all_vals.append( image[x][y] )
        all_vals = sorted(all_vals)
        thresh = 2 * (all_vals[-1] - all_vals[0] ) / 3.0

        row_labels = range(self.__xLimit)
        col_labels = range(self.__yLimit)
        cax = plt.matshow(image, cmap="gray")
        plt.colorbar(cax)
        plt.xticks(range(self.__yLimit), col_labels)
        plt.yticks(range(self.__xLimit), row_labels)

        for (j,i),label in np.ndenumerate(image):
            if label>thresh:
                plt.text(i,j,'{0:.2f}'.format(label), ha='center',va='center', color='black')
            else:
                plt.text(i,j,'{0:.2f}'.format(label), ha='center',va='center', color='white')

        #plt.legend()
        plt.show()



rnd = np.random.rand(10,10)
test = MetricGrid(10, 10, rnd)

test.showGridPlotValues()


