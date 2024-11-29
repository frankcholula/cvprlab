import numpy as np

def make_move(pt, movecode):
    """
    Make a move in the specified direction.
    
    Args:
        pt: Current point [x, y]
        movecode: Direction code (0-7)
            0: North     (0, -1)
            1: Northeast (1, -1)
            2: East      (1, 0)
            3: Southeast (1, 1)
            4: South     (0, 1)
            5: Southwest (-1, 1)
            6: West      (-1, 0)
            7: Northwest (-1, -1)
    
    Returns:
        list: New point coordinates [x, y]
    """
    moves = {
        0: (0, -1),   # N
        1: (1, -1),   # NE
        2: (1, 0),    # E
        3: (1, 1),    # SE
        4: (0, 1),    # S
        5: (-1, 1),   # SW
        6: (-1, 0),   # W
        7: (-1, -1)   # NW
    }
    dx, dy = moves[movecode]
    return [pt[0] + dx, pt[1] + dy]


def count_pixel_neighbourhood(mask, pt):
    """Count filled pixels in 8-neighborhood"""
    y, x = int(pt[1]), int(pt[0])
    y_min = max(0, y-1)
    y_max = min(mask.shape[0], y+2)
    x_min = max(0, x-1)
    x_max = min(mask.shape[1], x+2)
    wnd = mask[y_min:y_max, x_min:x_max]
    return np.sum(wnd) - 1

def make_move(pt, movecode):
    """Make a move in the specified direction"""
    moves = {
        0: (0, -1),   # N
        1: (1, -1),   # NE
        2: (1, 0),    # E
        3: (1, 1),    # SE
        4: (0, 1),    # S
        5: (-1, 1),   # SW
        6: (-1, 0),   # W
        7: (-1, -1)   # NW
    }
    dx, dy = moves[movecode]
    return [pt[0] + dx, pt[1] + dy]

def valid_move(mask, old_pt, movecode):
    """Check if move is valid"""
    pt = make_move(old_pt, movecode)
    if (pt[0] < 0 or pt[1] < 0 or 
        pt[0] >= mask.shape[1] or pt[1] >= mask.shape[0]):
        return False
    
    n = count_pixel_neighbourhood(mask, pt)
    return mask[int(pt[1]), int(pt[0])] == 1 and n < 8

def chaincode(mask):
    """
    Convert binary mask to chain code.
    
    Args:
        mask (np.ndarray): Binary mask of shape
        
    Returns:
        tuple: (chain_code, start_point)
            - chain_code: List of direction codes
            - start_point: Starting coordinates [x, y]
    """
    if not mask.any():
        raise ValueError("Empty mask provided")
    
    mask = (mask > 0).astype(float)
    VISITED = 2
    
    # Find starting point
    rows, cols = np.where(mask)
    startpoint = None
    
    for i in range(len(rows)):
        if count_pixel_neighbourhood(mask, [cols[i], rows[i]]) < 8:
            startpoint = [cols[i], rows[i]]
            break
    
    if startpoint is None:
        raise ValueError("Could not find valid starting point")
    
    # Visit start point
    mask[int(startpoint[1]), int(startpoint[0])] = VISITED
    cc = []
    MOVES = [0, 4, 2, 6, 1, 5, 3, 7]  # NE SW SE NW N S E W
    moved = True
    current_point = startpoint
    
    while moved:
        moved = False
        for m in MOVES:
            if valid_move(mask, current_point, m):
                cc.append(m)
                current_point = make_move(current_point, m)
                mask[int(current_point[1]), int(current_point[0])] = VISITED
                moved = True
                break
    
    return cc, np.array(startpoint)
