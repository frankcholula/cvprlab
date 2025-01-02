"""
Main script for the Visual Hull lab.
"""

from .camera import load_camera
from .visualization import plot_cameras
from .reconstruction import generate_masks, reconstruct_visual_hull

def main():
    """Main function implementing the lab workflow"""
    print("\nEEE3032 Visual Hull Lab")
    print("======================")
    
    # Configuration
    bounds = {
        'x': (0, 4),
        'y': (0, 2),
        'z': (0, 4)
    }
    steps = (0.05, 0.05, 0.05)
    threshold = 5
    num_cameras = 8
    
    try:
        # Ex1: Load and visualize cameras
        print("\nEx1: Loading camera calibration...")
        cameras = load_camera()
        if not cameras:
            return
        
        print("Visualizing camera positions...")
        plot_cameras(cameras)
        
        # Ex2: Generate masks
        print("\nEx2: Generating masks from images...")
        masks = generate_masks(num_cameras)
        
        # Ex3: Reconstruct visual hull
        print("\nEx3: Reconstructing visual hull...")
        visual_hull = reconstruct_visual_hull(cameras, masks, bounds, steps, threshold)
        
    except Exception as e:
        print(f"Error in lab execution: {str(e)}")
        return

if __name__ == "__main__":
    main()
