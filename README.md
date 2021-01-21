# Implementing a Clustering Algorthim using PySpark

**Spark is a general-purpose, in-memory, distributed processing engine that allows you to process your data efficiently in a distributed fashion.**

I will be implementing K-means clustering algorithms on PySpark. Will use Euclidean distance, Manhattan distance as cost functions to minimize when we assign points to clusters and comapre them.

```data.txt``` contains the dataset which has 4601 rows and 58 columns. Each row is a document represented as a 58-dimensional vector of features. Each component in the vector represents the importance of a word in the document.

```c1.txt``` contains k initial cluster centroids. These centroids were chosen by selecting k = 10 random points from the input data.

```c2.txt``` contains initial cluster centroids which are as far apart as possible.

Setting the number of iterations to 20 and the number of clusters k to 10.

Here is what I have explored

**EUCLIDEAN DISTANCE & DIFFERENT INITAL CLUSTER CENTROIDS**

Looking at the graphs from two different files (*c1.txt*, *c2.txt*), we see that the cost decreases. Still, looking at the cost's values, we can say that the centroids with farther distance possible data have started with less cost than random centroids.

![Euclidean C1](https://github.com/DVD-99/Clustering-using-PySpark/blob/main/imgs/euc_c1.png)
![Euclidean C2](https://github.com/DVD-99/Clustering-using-PySpark/blob/main/imgs/euc_c2.png)



**MANHATTAN DISTANCE & DIFFERENT INITAL CLUSTER CENTROIDS**

I say that it is opposite to that of Euclidean distance. The random centroids have less cost than the centroids that are farthest possible. You can notice the sharp decrease in the cost from the first iteration to the second and then gain few points and finally decreases.

![Manhattan C1](https://github.com/DVD-99/Clustering-using-PySpark/blob/main/imgs/man_c1.png)
![Manhattan C2](https://github.com/DVD-99/Clustering-using-PySpark/blob/main/imgs/man_c1.png)

