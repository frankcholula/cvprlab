import numpy as np

class EigenModel:
    """
    Class to hold eigenmodel data and methods for shape analysis.
    
    Attributes:
        N (int): Number of observations used to build the model
        D (int): Dimension of each observation
        org (np.ndarray): Mean of observations (d x 1 matrix)
        vct (np.ndarray): Matrix of eigenvectors (one per column)
        val (np.ndarray): Vector of eigenvalues (matching vct columns)
    """
    def __init__(self):
        self.N = 0          # Number of observations
        self.D = 0          # Dimension of observations
        self.org = None     # Mean vector
        self.vct = None     # Eigenvectors
        self.val = None     # Eigenvalues

def eigen_build(obs):
    """
    Build an eigenmodel from a set of observations.
    
    This function:
    1. Computes the mean of observations
    2. Centers the data by subtracting the mean
    3. Computes the covariance matrix
    4. Finds eigenvectors and eigenvalues
    5. Sorts them by eigenvalue magnitude
    
    Args:
        obs (np.ndarray): d x n matrix where:
            d = dimension of each observation
            n = number of observations
            
    Returns:
        EigenModel: Constructed eigenmodel with:
            - Mean vector (org)
            - Eigenvectors (vct)
            - Eigenvalues (val)
            
    Raises:
        ValueError: If observations matrix is empty or improperly shaped
    """
    # Input validation
    if obs is None or obs.size == 0:
        raise ValueError("Empty observation set")
    
    if len(obs.shape) != 2:
        raise ValueError("Observations must be a 2D matrix")
        
    # Create new eigenmodel
    E = EigenModel()
    
    # Store dimensions
    E.N = obs.shape[1]  # Number of observations
    E.D = obs.shape[0]  # Dimension of each observation
    
    # Compute mean of observations
    E.org = np.mean(obs, axis=1, keepdims=True)
    
    # Center the observations by subtracting mean
    obs_translated = obs - E.org
    
    # Compute covariance matrix
    # Factor of 1/N gives us the average outer product
    C = (1/E.N) * (obs_translated @ obs_translated.T)
    
    # Compute eigenvalues and eigenvectors
    # Using eigh since covariance matrix is symmetric
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort eigenvectors and eigenvalues by eigenvalue magnitude (descending)
    idx = eigenvalues.argsort()[::-1]
    E.val = eigenvalues[idx]
    E.vct = eigenvectors[:, idx]
    
    return E

def eigen_deflate(E, method, param):
    """
    Reduce the dimensionality of an eigenmodel by keeping only significant components.
    
    Two methods available:
    1. 'keepn': Keep the n most significant eigenvectors
    2. 'keepf': Keep enough eigenvectors to retain fraction f of total energy
    
    Args:
        E (EigenModel): Eigenmodel to deflate
        method (str): Either 'keepn' or 'keepf'
        param (float): 
            - If method='keepn': number of eigenvectors to keep
            - If method='keepf': fraction of energy to retain (0 to 1)
            
    Returns:
        EigenModel: Deflated eigenmodel
        
    Raises:
        ValueError: If method is invalid or parameters are out of range
    """
    if not isinstance(E, EigenModel):
        raise ValueError("First argument must be an EigenModel")
        
    if method not in ['keepn', 'keepf']:
        raise ValueError("Method must be 'keepn' or 'keepf'")
    
    if method == 'keepn':
        # Validate parameter
        if not isinstance(param, (int, np.integer)) or param < 1:
            raise ValueError("For 'keepn', param must be positive integer")
        if param > E.vct.shape[1]:
            raise ValueError(f"Cannot keep {param} components; only {E.vct.shape[1]} available")
            
        # Keep top n components
        E.val = E.val[:param]
        E.vct = E.vct[:, :param]
        
    elif method == 'keepf':
        # Validate parameter
        if not 0 < param <= 1:
            raise ValueError("For 'keepf', param must be between 0 and 1")
            
        # Compute total energy (sum of eigenvalues)
        total_energy = np.sum(np.abs(E.val))
        current_energy = 0
        rank = 0
        
        # Keep adding eigenvalues until we reach desired energy fraction
        for i in range(E.vct.shape[1]):
            if current_energy > (total_energy * param):
                break
            rank += 1
            current_energy += E.val[i]
        
        # Keep components up to computed rank
        E.val = E.val[:rank]
        E.vct = E.vct[:, :rank]
    
    return E

def eigen_mahalanobis(obs, E):
    """
    Compute Mahalanobis distance between observations and eigenmodel.
    
    This measures how many standard deviations each observation is from
    the eigenmodel's mean, taking into account the covariance structure.
    
    The computation:
    1. Centers observations by subtracting eigenmodel mean
    2. Projects centered observations onto eigenvectors
    3. Normalizes by eigenvalues (which represent variance in each direction)
    4. Computes total distance as sum of normalized squared distances
    
    Args:
        obs (np.ndarray): d x n matrix of observations to measure
        E (EigenModel): Eigenmodel to measure distance from
        
    Returns:
        np.ndarray: Vector of distances (one per observation)
        
    Raises:
        ValueError: If dimensions don't match or inputs are invalid
    """
    # Input validation
    if not isinstance(E, EigenModel):
        raise ValueError("Second argument must be an EigenModel")
        
    if obs is None or obs.size == 0:
        raise ValueError("Empty observation set")
        
    if obs.shape[0] != E.vct.shape[0]:
        raise ValueError(f"Observation dimension ({obs.shape[0]}) must match eigenmodel ({E.vct.shape[0]})")
    
    # Center the observations
    obs_translated = obs - E.org
    
    # Project onto eigenvectors
    proj = E.vct.T @ obs_translated
    
    # Compute squared distances
    dist_sq = proj * proj
    
    # Handle zero eigenvalues to avoid division by zero
    # This effectively ignores components in directions of zero variance
    safe_val = E.val.copy()
    safe_val[safe_val == 0] = 1
    
    # Normalize by eigenvalues and sum
    dist = dist_sq / safe_val.reshape(-1, 1)
    
    # Return square root of sum (Mahalanobis distance)
    return np.sqrt(np.sum(dist, axis=0))

# Optional: Example usage and testing
if __name__ == "__main__":
    # Create some sample 2D data
    np.random.seed(42)
    n_samples = 100
    data = np.random.randn(2, n_samples)  # 2D Gaussian data
    
    # Build eigenmodel
    E = eigen_build(data)
    print("Original eigenmodel dimensions:", E.vct.shape)
    
    # Deflate using both methods
    E_keepn = eigen_deflate(E, 'keepn', 1)
    print("After keepn deflation:", E_keepn.vct.shape)
    
    E_keepf = eigen_deflate(E, 'keepf', 0.95)
    print("After keepf deflation:", E_keepf.vct.shape)
    
    # Compute distances to some test points
    test_points = np.random.randn(2, 5)
    distances = eigen_mahalanobis(test_points, E)
    print("Distances to test points:", distances)