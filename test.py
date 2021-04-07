import numpy as np

def euclideanDistance(point, centroid):
    """
    Function implementing the euclidean distance between two points

    :param point
        np.array of a data point [x,y]

    :param centroid 
        np.array of centroid [x,y]
    
    :return 
        euclidean distance between centroid and point
    """
    return np.sqrt(np.sum((point - centroid)**2))

def calculateCentroid(clusters):
    """
    Function implementing the new centroids 

    :param clusters
        a list with all points in one cluster

    :param centroids 
        a numpy array of the centroids 

    :return 
        a numpy array of the updated centroids 
    """
    updated_centroids = []
    
    for cluster in clusters:
        # Finding the average point given all points in one cluster
        calculate_new_centroid = 1.0 * np.mean(cluster, axis=0) 
        updated_centroids.append(calculate_new_centroid)
    return updated_centroids

def kmeans(data, centroids):
    """
    Function implementing the k-means clustering.
    
    :param data
        a 2D array with points
    :param centroids
        initial centroids
        a 2D array containing the coordinates of initial points
    :return
        final centroids
    """
    clusters = []

    for i in range(len(centroids)): 
        clusters.append([])
    # For each point in data array
    for point in data: 
        # Find the distance between a row (point) and the centroids, key is centroid and item is distance between
        distances = []
        for centroid in centroids:
            # Find the distances
            distances.append(euclideanDistance(point, centroid))
        # Placing a point into a cluster given the index of the minimum value
        min_dist_index = distances.index(min(distances))
        clusters[min_dist_index].append(point)
    
    # Recalculate the centroids 
    new_centroids = calculateCentroid(clusters)

    # Deciding when to stop calulating the new centroids and when not to
    if (np.array_equal(new_centroids,centroids)):
        return new_centroids,clusters
    else:
        kmeans(data,new_centroids)
        return new_centroids, clusters

def calculateDistanceMatrix(data):
    distance_matrix = np.zeros((len(data), len(data)))

    # Only gets the lower bottom of the distance matrix, which hehe works for this case i guess :3
    # Stonks saving computing time 
    for row in range(len(distance_matrix)):
        for col in range(row):
            euclid_dist = euclideanDistance(data[row], data[col]) 
            distance_matrix[row,col] = euclid_dist
            distance_matrix[col,row] = euclid_dist
    
    return distance_matrix

def average(lst):
    return sum(lst) / len(lst)

def calculateSeparation(cluster, clusters):
    """
    Function implementing the calculation of separation
    This part will consist of calculating the average distance points in the nearest cluster
    Given a point which cluster is the nearest
    
    """
    b_cluster = []
    temp_clusters = clusters.copy()
    temp_clusters.remove(cluster)

    # The implementation of the clusters were done in a fucking stupid way 
    # Rip spaghetti code 
    for i in cluster:
        avg_dist = []
        for j in temp_clusters:
            distances = []
            for point in j:
                distances.append(euclideanDistance(i,point))
            avg_dist.append(average(distances))
        shortest_distance_cluster = min(avg_dist)
        b_cluster.append(shortest_distance_cluster)
    return b_cluster

def calculateCohesion(cluster):
    """
    Function implementing the calculation of cohesion
    Calculate avg distance of points in the nearest cluster, then take the min of all those values
    
    :param cluster
        A np.array of a cluster
    :param distance_matrix
        A matrix with all the distances
    :param point
        A point in the cluster
    :return
        a list of mean values of the distance between all the points in the cluster
    """
    distance_matrix = calculateDistanceMatrix(cluster)

    #Summing all columns in distance matrix given a cluster
    a = np.sum(distance_matrix, axis=0)
    # Finding average of all points except the one in the matrix which is zero
    a = a/(len(a)-1)
    
    return a

def silhouette_score(data, clusters):
    """
    Function implementing the k-means clustering.
    
    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    # List of the average distances from all points given a cluster
    a_clusters = []
    b_clusters = []

    # Creating distance matrix for all of the data 
    distance_matrix = calculateDistanceMatrix(data)

    for cluster in clusters:
        a = calculateCohesion(cluster)
        a_clusters.append(a)
        b = calculateSeparation(cluster, clusters)
        b_clusters.append(b)
        


    score = (b-a)/max(b-a)
    
    return score

data = np.array([
    [1,4],
    [2,4],
    [2,5],
    [3,4],
    [4,1],
    [4,3],
    [5,1],
    [5,2],
    [6,1],
    [3,3],
    [4,4],
    [4,5],
    [5,5],
    [6,6]
])

# Får ut array med clusterne
clusters = [[np.array([1,4]), np.array([2,4]), np.array([2,5])],
            [np.array([3,4]),np.array([4,1]),np.array([4,3]),np.array([5,1]),np.array([5,2]),np.array([6,1])],
            [np.array([3,3]),np.array([4,4]),np.array([4,5]),np.array([5,5]),np.array([6,6])],
            ]
""" ------- TESTER -------- """

def testDistanceMatrix(data):
    distance_matrix = calculateDistanceMatrix(data)

def testSilhoutteScore(data,clusters):
    silhouette = silhouette_score(data,clusters)

testSilhoutteScore(data,clusters)

