import numpy as np

def sample_polygon_perimeter(p, s):
    """
    Sample points along polygon perimeter uniformly.
    
    Args:
        p (np.ndarray): Polygon vertices [x1 x2..;y1 y2..]
        s (int): Number of samples
        
    Returns:
        np.ndarray: Sampled points [x1 x2..;y1 y2..]
    """
    if p.shape[1] < 2:
        raise ValueError("Polygon must have at least two vertices")
    
    # Compute total polygon length
    v = np.hstack((p[:, 1:], p[:, 0:1])) - p
    total_len = np.sum(np.sqrt(np.sum(v**2, axis=0)))
    
    # Compute step size
    rate = total_len / s
    
    # Sample points
    points = []
    current_len = 0
    current_pos = p[:, 0:1]
    next_vertex_index = 1
    
    while current_len < total_len and len(points) < s:
        v = p[:, next_vertex_index:next_vertex_index+1] - current_pos
        dst_left = np.linalg.norm(v)
        
        if dst_left > rate:
            # Move along current edge
            v = v / np.linalg.norm(v)
            current_pos = current_pos + v * rate
        else:
            # Move to next vertex
            dst_left = rate - dst_left
            while dst_left > 0:
                current_pos = p[:, next_vertex_index:next_vertex_index+1]
                next_vertex_index = (next_vertex_index + 1) % p.shape[1]
                v = p[:, next_vertex_index:next_vertex_index+1] - current_pos
                this_adv = min(dst_left, np.linalg.norm(v))
                v = v / np.linalg.norm(v)
                current_pos = current_pos + v * this_adv
                dst_left = dst_left - this_adv
        
        current_len += rate
        points.append(current_pos.flatten())
    
    return np.array(points).T