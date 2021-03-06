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

    :param cluster
        List of a cluster we are looking at + its point
    :param clusters
        List with all of the clusters
    :return b_cluster
        Returns an array with b-values for points in :param cluster
    """
    b_cluster = []

    # Removing cluster we are looking at from clusters
    temp_clusters = clusters.copy()
    count = 0
    for temp_cluster in temp_clusters:
        if np.sum(temp_cluster) == np.sum(cluster):
            temp_clusters.pop(count)
        count =+ 1


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
        A list of mean values of the distance between all the points in the cluster
    """
    distance_matrix = calculateDistanceMatrix(cluster)

    #Summing all columns in distance matrix given a cluster
    a = np.sum(distance_matrix, axis=0)
    # Finding average of all points except the one in the matrix which is zero
    a = a/(len(a)-1)
    
    return a

def silhouetteScore(data, clusters):
    """
    Function implementing the k-means clustering.
    
    :param data
        data
    :param centroids
        centroids
    :return
        mean Silhouette Coefficient of all samples
    """
    # List of all silhouette scores for each point given a cluster
    silhoutte_scores_cluster = []

    for cluster in clusters:
        a = calculateCohesion(cluster)
        b = calculateSeparation(cluster, clusters)
        temp_silhoutte_clusters = []
        for i in range(len(a)):
            numerator = b[i]-a[i]
            denominator = max(a[i],b[i])
            sil = numerator/denominator
            temp_silhoutte_clusters.append(sil)
        silhoutte_scores_cluster.append(average(temp_silhoutte_clusters))
    silhouette_score = average(silhoutte_scores_cluster)
    return silhouette_score


""" ------- TESTER -------- """
test_data = np.array([
    [66.24345364, 57.31053969],
    [43.88243586, 39.69929645],
    [44.71828248, 48.38791398],
    [39.27031378, 48.07972823],
    [58.65407629, 55.66884721],
    [26.98461303, 44.50054366],
    [67.44811764, 49.13785896],
    [42.38793099, 45.61070791],
    [53.19039096, 50.21106873],
    [47.50629625, 52.91407607],
    [2.29566576, 20.15837474],
    [18.01306597, 22.22272531],
    [16.31113504, 20.1897911 ],
    [13.51746037, 19.08356051],
    [16.30599164, 20.30127708],
    [5.21390499, 24.91134781],
    [9.13976842, 17.17882756],
    [3.44961396, 26.64090988],
    [8.12478344, 36.61861524],
    [13.71248827, 30.19430912],
    [74.04082224, 23.0017032 ],
    [70.56185518, 16.47750154],
    [71.26420853, 8.57481802],
    [83.46227301, 16.50657278],
    [75.25403877, 17.91105767],
    [71.81502177, 25.86623191],
    [75.95457742, 28.38983414],
    [85.50127568, 29.31102081],
    [75.60079476, 22.85587325],
    [78.08601555, 28.85141164]
])
test_centroids = np.array([
    [25, 50],
    [50, 50],
    [75, 50]
])

test_centroids = np.asarray(kmeans(test_data, test_centroids)[0])

clusters = kmeans(test_data,test_centroids)[1]
print(silhouetteScore(test_data, clusters))
