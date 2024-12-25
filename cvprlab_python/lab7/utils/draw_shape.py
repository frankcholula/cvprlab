import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def draw_shape(canvas_size):
    """
    Interactive shape drawing function.
    
    Args:
        canvas_size (int): Size of the square canvas
        
    Returns:
        tuple: (mask, points)
            - mask: Binary mask of the drawn shape
            - points: Array of vertex coordinates [x; y]
    """
    fig, ax = plt.subplots()
    ax.set_title('Click to draw a polygon (right click = last point)')
    
    # Create white canvas
    canvas = np.ones((canvas_size, canvas_size, 3))
    ax.imshow(canvas)
    ax.set_aspect('equal')    
    # Initialize points list
    pts = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        if event.button == 1:  # Left click
            pts.append([event.xdata, event.ydata])
            # Redraw
            ax.clear()
            ax.imshow(canvas)
            points = np.array(pts)
            if len(points) > 0:
                ax.plot(points[:, 0], points[:, 1], 'b*-')
                ax.plot(points[-1, 0], points[-1, 1], 'ro')
            plt.draw()
        
        elif event.button == 3 and len(pts) > 2:  # Right click
            pts.append([event.xdata, event.ydata])
            # Complete polygon
            points = np.array(pts)
            ax.plot(points[:, 0], points[:, 1], 'b*-')
            ax.plot([points[-1, 0], points[0, 0]], 
                   [points[-1, 1], points[0, 1]], 'b-')
            plt.draw()
            plt.close()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    if len(pts) < 3:
        raise ValueError("At least 3 points needed to form a shape")
    
    # Convert to numpy array and reshape
    points = np.array(pts).T
    
    # Create binary mask
    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
    y, x = np.mgrid[:canvas_size, :canvas_size]
    points_path = Path(points.T)
    mask = points_path.contains_points(np.vstack((x.flatten(), y.flatten())).T).reshape(canvas_size, canvas_size)
    
    return mask, points
