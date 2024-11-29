import numpy as np
import matplotlib.pyplot as plt 
from .chaincode import make_move 

def chaincode_rasterize(cc, canvas_size=None, startpoint=None):
    """
    Rasterize chain code to binary mask and polygon.
    
    Args:
        cc: Chain code sequence
        canvas_size: Optional size of output canvas
        startpoint: Optional starting point
        
    Returns:
        tuple: (mask, polygon_points)
    """
    if not cc:
        raise ValueError("Empty chain code provided")
    
    if startpoint is None:
        startpoint = np.array([0, 0])
    
    if len(startpoint.shape) > 1 and startpoint.shape[0] == 1:
        startpoint = startpoint.T
    
    # Follow chain code to get polygon points
    pts = [startpoint]
    current_point = startpoint.copy()
    
    for c in cc:
        current_point = make_move(current_point, c)  # Using imported make_move
        pts.append(current_point)
    
    pts = np.array(pts).T
    
    # Handle canvas size
    if canvas_size is None:
        pts = np.round(pts)
        pts = pts - np.array([[np.min(pts[0])], [np.min(pts[1])]])
        canvas = np.zeros((int(np.max(pts[1])) + 1, int(np.max(pts[0])) + 1))
    else:
        canvas = np.zeros(canvas_size)
    
    # Create binary mask
    y, x = np.mgrid[:canvas.shape[0], :canvas.shape[1]]
    points_path = plt.Path(pts.T)
    mask = points_path.contains_points(np.vstack((x.flatten(), y.flatten())).T).reshape(canvas.shape)
    
    return mask, pts
