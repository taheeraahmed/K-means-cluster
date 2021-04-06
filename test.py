import numpy as np
#%matplotlib inline
from matplotlib import pyplot as plt

def plot_clusters(data, centroids):
    """
    Shows a scatter plot with the data points clustered according to the centroids.
    """
    # Assigning the data points to clusters/centroids.
    clusters = [[] for _ in range(centroids.shape[0])]
    for i in range(data.shape[0]):
        distances = np.linalg.norm(data[i] - centroids, axis=1)
        clusters[np.argmin(distances)].append(data[i])

    # Plotting clusters and centroids.
    fig, ax = plt.subplots()
    for c in range(centroids.shape[0]):
        if len(clusters[c]) > 0:
            cluster = np.array(clusters[c])
            ax.scatter(cluster[:, 0], cluster[:, 1], s=7)
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red')

    # We would like to have some control over the randomly generated data.
# This is just for development purposes.
np.random.seed(0)

# Euclidean space.
DIMENSIONS = 2

# We will generate clusters.
CLUSTERS = [
    {
        'mean': (10, 10),
        'std': (10, 5),
        'size': 300
    },
    {
        'mean': (10, 85),
        'std': (10, 3),
        'size': 100
    },
    {
        'mean': (50, 50),
        'std': (6, 6),
        'size': 200
    },
    {
        'mean': (80, 75),
        'std': (5, 10),
        'size': 200
    },
    {
        'mean': (80, 20),
        'std': (5, 5),
        'size': 100
    }
]

# Initializing the dataset with zeros.
synthetic_data = np.zeros((np.sum([c['size'] for c in CLUSTERS]), DIMENSIONS))

# Generating the clusters.
start = 0
for c in CLUSTERS:
    for d in range(DIMENSIONS):
        synthetic_data[start:start + c['size'], d] = np.random.normal(c['mean'][d], c['std'][d], (c['size']))
    start += c['size']

plt.figure()
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], s=3)

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

def calculateCohesion(cluster,distance_matrix):
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

    for cluster in clusters:
        for point in cluster:
            a = calculateCohesion(cluster, distance_matrix)
            b = calculateSeparation(cluster)
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
testDistanceMatrix(data)

