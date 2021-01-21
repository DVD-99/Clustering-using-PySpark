'''
Written by Vishnu Dutt Duggirala
CSCI 5702 Big Data Mining
Assignment-2 Part-1
Turn in date - 12/25/19
'''

import numpy as np
from numpy import linalg as LA
from pyspark import SparkContext, SparkConf
from scipy.spatial import distance
import matplotlib.pyplot as plt

#starting a spark session

conf = SparkConf()
sc = SparkContext().getOrCreate(conf = conf)


#cost function caluclates the cost for euclidean and manhattan
def cost(data, eucli):  #if true finds the cost of euclidean
    if eucli is True:
        cos = data.map(lambda w: w[1]**2)
        co = cos.map(lambda w: np.min(w))
        return sum(co.collect())
    else:
        cos = data.map(lambda w: w[1])
        co = cos.map(lambda w: np.min(w))
        return sum(co.collect())


#function for kmeans 

def kmeans(data, centroids, iterations, euclidean_distance):
    cost_list = []
    for i in range(iterations):
        if euclidean_distance is True:
            #adding the distance from the centroids into a numpy array and mapping with each row 
            qwerty = data.map(lambda w: (w, np.array([distance.euclidean(w,c) for c in centroids])))
            cost_list.append(cost(qwerty,euclidean_distance))   #adding all the cost into a list
        else:
            qwerty = data.map(lambda w: (w, np.array([distance.cityblock(w,c) for c in centroids])))
            cost_list.append(cost(qwerty,euclidean_distance))
        print("Iteration: ", i)
        #adding the index of the min distance
        cen = qwerty.map(lambda w: (np.argmin(w[1]), w))
        #removed the distances from the rdd
        ce = cen.map(lambda w : (w[0],w[1][0]))
        #grouping and sorting by the key which is index of the centroid 
        c = ce.groupByKey().sortByKey().mapValues(list)
        #finding the average which is new centroid and replacing the old ones
        centroid = c.map(lambda w: np.average(w[1],axis = 0))
        centroids = np.array(centroid.collect())
    return cost_list


# reading the data.txt into a rdd
dataset = sc.textFile("data.txt").map(lambda v: v.split(' '))
data = dataset.map(lambda w: np.array([(float(c)) for c in w]))
#reading the c1.txt into a numpy array
c1 = np.loadtxt("c1.txt")
c_e = kmeans(data, c1, 20, True)
c_m = kmeans(data, c1, 20, False)



#plotting the Cost vs Iteration for c1.txt(Euclidean)
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
plt.plot(x,c_e)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Iteration')
plt.ylabel('Cost of Euclidean distance')
plt.title('Cost vs Iteration for c1.txt')
plt.show()


#Manhattan distance
plt.plot(x,c_m)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Iteration')
plt.ylabel('Cost of Manhattan distance')
plt.title('Cost vs Iteration for c1.txt')
plt.show()


#reading the c2.txt into numpy array
c2 = np.loadtxt("c2.txt")
c_e_2 = kmeans(data, c2, 20, True)
c_m_2 = kmeans(data, c2, 20, False)


#plotting the Cost vs Iteration for c2.txt(Euclidean)
plt.plot(x,c_e_2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Iteration')
plt.ylabel('Cost of Euclidean distance')
plt.title('Cost vs Iteration for c2.txt')
plt.show()

#Manhattan distance
plt.plot(x,c_m_2)
plt.xticks(np.arange(0, 20, 1))
plt.xlabel('Iteration')
plt.ylabel('Cost of Manhattan distance')
plt.title('Cost vs Iteration for c2.txt')
plt.show()

#stoping the spark session
sc.stop()

