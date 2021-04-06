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
    # The key of the dictionary will be the centroids, while the items will be the points in the data
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
            distance_matrix[row,col] = euclideanDistance(data[row], data[col]) 

def calculateSeparation(cluster):
    return 0

def calculateCohesion(cluster,distance_matrix,point):
    """
    Function implementing the calculation of cohesion
    Given a cluster and a point: it should calculate the average distance between all points in the cluster. 
    
    :param cluster
        A np.array of a cluster
    :param distance_matrix
        A matrix with all the distances
    :param point
        A point in the cluster
    :return
        mean value of the distance between all the points in the cluster
    """


    return 0

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
    distance_matrix = calculateDistanceMatrix(data)
    a = []
    b = []

    for cluster in clusters:
        for point in cluster:
            a_value = calculateCohesion(cluster, distance_matrix, point)
            a.append(a_value)
            #b = calculateSeparation(cluster)
    # Step 2: For each data point
    ## a: Calculate cohesion
    ## b: Calculate separation
    ## S = (b-a)/max(b-a)
    score = (b-a)/max(b-a)
    
    return score

data = np.array([
    [1, 2],
    [2, 2],
    [3, 2],
    [5, 1],
    [3, 0],
    [2, 4],
    [4, 2],
])

centroids = np.array([
    [1, 2],
    [2, 2]
])

# Får ut array med clusterne
clusters = np.asarray(kmeans(data,centroids)[1])

""" ------- TESTER -------- """

def testDistanceMatrix(data):
    distance_matrix = calculateDistanceMatrix(data)

def testSilhoutteScore(data,clusters):
    silhouette = silhouette_score(data,clusters)

testSilhoutteScore(data,clusters)

