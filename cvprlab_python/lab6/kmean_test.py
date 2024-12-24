import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Parameters
RANGE = 100
NCLUSTERS = 5
SPREAD = 5
NPOINTS = 100
DIMENSION = 2  # Plotting will only work if this is 2

# 1) Generate NCLUSTERS of synthetic data
data = []
for n in range(NCLUSTERS):
    cluster_mean = (np.random.rand(DIMENSION) - np.random.rand(DIMENSION)) * RANGE
    cluster_data = np.random.randn(DIMENSION, NPOINTS) * SPREAD + cluster_mean[:, np.newaxis]
    data.append(cluster_data)

# 2) Plot data - this is the "ground truth" we hope k-means can recover
plt.figure()
plt.title('Synthetic data (ground truth)')
plt.xlim(-RANGE, RANGE)
plt.ylim(-RANGE, RANGE)
for cluster_data in data:
    plt.scatter(cluster_data[0, :], cluster_data[1, :], marker='x')

plt.show()

# 3) Concatenate all the data
alldata = np.hstack(data)

# 4) Run KMeans and plot the centers of the identified clusters
print(f'Running KMeans over {alldata.shape[1]} points (of dimension {DIMENSION})')

# Random initial cluster centers
starting_centres = np.random.rand(NCLUSTERS, DIMENSION) * RANGE - (RANGE / 2)

# Implementing KMeans manually (you can use sklearn's KMeans for simplicity)
def kmeans(data, k, init_centres, max_iters=100):
    centres = init_centres.copy()
    for _ in range(max_iters):
        # Assign points to nearest centre
        distances = cdist(data.T, centres)
        labels = np.argmin(distances, axis=1)
        
        # Update centres
        new_centres = np.array([data[:, labels == i].mean(axis=1) if np.any(labels == i) else centres[i] for i in range(k)])
        
        # Check for convergence
        if np.allclose(new_centres, centres):
            break
        centres = new_centres
    return centres, labels

centres, labels = kmeans(alldata, NCLUSTERS, starting_centres)

plt.figure()
plt.title('Clustered data after KMeans')
plt.xlim(-RANGE, RANGE)
plt.ylim(-RANGE, RANGE)

colours = np.random.rand(NCLUSTERS, 3)
for i, point in enumerate(alldata.T):
    plt.scatter(point[0], point[1], color=colours[labels[i]], marker='x')

# Plot the cluster centers
plt.scatter(centres[:, 0], centres[:, 1], color='cyan', marker='*', s=200, label='Centres')
plt.legend()
plt.show()

# 5) Make a nearest neighbour assignment (manual KNN-like step)
all_dists = np.array([np.linalg.norm(alldata - centre[:, np.newaxis], axis=0) for centre in centres])
classification = np.argmin(all_dists, axis=0)

# 6) Plot the assignments
plt.figure()
plt.title('Clustered data after KMeans/KNN')
plt.xlim(-RANGE, RANGE)
plt.ylim(-RANGE, RANGE)

for i, point in enumerate(alldata.T):
    plt.scatter(point[0], point[1], color=colours[classification[i]], marker='x')

plt.show()

# 7) Check the ground truth
groundtruth = np.ones(NPOINTS * NCLUSTERS)  # Array of same length as all points
for i in range(NCLUSTERS):
   groundtruth[i*NPOINTS:(i+1)*NPOINTS] = i  # Assign cluster number i to each group of points

matches = np.array([[np.sum((groundtruth == i) & (classification == j)) 
                   for j in range(NCLUSTERS)] 
                  for i in range(NCLUSTERS)])

correct_matches = matches.max(axis=1).sum()
print(f'Number of correct classifications: {correct_matches} / {len(groundtruth)}')