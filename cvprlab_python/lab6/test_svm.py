import numpy as np
from sklearn.svm import SVC

# Parameters
SPLIT_FOR_TRAINING = 0.2
RANGE = 100
N_CLUSTERS = 3
SPREAD = 5
N_POINTS = 100
DIMENSION = 2

# 1) Generate N_CLUSTERS of synthetic data
data = []
for n in range(N_CLUSTERS):
    cluster_mean = (np.random.rand(DIMENSION) - np.random.rand(DIMENSION)) * RANGE
    cluster_data = np.random.randn(N_POINTS, DIMENSION) * SPREAD + cluster_mean
    data.append(cluster_data)

# 2) Separate data into training and test sets
cutoff = round(N_POINTS * SPLIT_FOR_TRAINING)
train_data = []
test_data = []
for cluster in data:
    train_data.append(cluster[:cutoff])
    test_data.append(cluster[cutoff:])

# Combine all training and test data
all_train_obs = np.vstack(train_data)
all_test_obs = np.vstack(test_data)

# Create class labels
all_train_class = np.hstack([np.ones(cutoff) * (i + 1) for i in range(N_CLUSTERS)])
all_test_class = np.hstack([np.ones(N_POINTS - cutoff) * (i + 1) for i in range(N_CLUSTERS)])

# 3) Setup SVM parameters and train
svm = SVC(kernel='rbf', 
          gamma=1/(2*5**2),  # equivalent to gaussian kernel with kerneloption=5
          C=10000000,
          tol=1e-7)

# 4) Train the SVM with training data
svm.fit(all_train_obs, all_train_class)

# Evaluate training accuracy
train_pred = svm.predict(all_train_obs)
train_accuracy = 100 * np.mean(train_pred == all_train_class)
print(f'\nClassification correct on training data: {train_accuracy:.2f}')

# 5) Test the classifier on test data
test_pred = svm.predict(all_test_obs)
test_accuracy = 100 * np.mean(test_pred == all_test_class)
print(f'\nClassification correct on test data: {test_accuracy:.2f}')

# Optional: Visualize results if DIMENSION == 2
if DIMENSION == 2:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    
    # Plot training data
    plt.subplot(121)
    for i in range(N_CLUSTERS):
        mask = all_train_class == i + 1
        plt.scatter(all_train_obs[mask, 0], all_train_obs[mask, 1], label=f'Cluster {i+1}')
    plt.title('Training Data')
    plt.legend()
    
    # Plot test data
    plt.subplot(122)
    for i in range(N_CLUSTERS):
        mask = all_test_class == i + 1
        plt.scatter(all_test_obs[mask, 0], all_test_obs[mask, 1], label=f'Cluster {i+1}')
    plt.title('Test Data')
    plt.legend()
    
    plt.tight_layout()
    plt.show()