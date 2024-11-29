"""
main.py
Main interface for CVPR Multiview Geometry Lab
"""

from exercises.fundamental import (
    exercise1_fundamental_matrix,
    exercise2_epipolar_geometry,
    exercise3_visual_verification,
    exercise4_mathematical_verification
)
from exercises.triangulation import (
    exercise5_triangulation,
    exercise6_multiple_points
)
from exercises.homography import (
    exercise7_homography_estimation,
    exercise8_homography_interactive
)

def print_header():
    """Print lab header information"""
    print("\nComputer Vision and Pattern Recognition Lab")
    print("Multiview Geometry Exercises")
    print("=" * 50)

def print_menu():
    """Print available exercises menu"""
    print("\nAvailable exercises:")
    print("1: Fundamental Matrix Estimation (8-point algorithm)")
    print("2: Understanding Epipolar Geometry")
    print("3: Visual Verification of Epipolar Lines")
    print("4: Mathematical Verification")
    print("5: 3D Point Triangulation")
    print("6: Multiple Point Triangulation")
    print("7: Homography Estimation")
    print("8: Interactive Homography Visualization")
    print("q: Quit")

def main():
    """Main lab interface"""
    print_header()
    
    # Store results that might be needed across exercises
    results = {
        'F': None,  # Fundamental matrix
        'points_left': None,
        'points_right': None,
        'H': None,  # Homography matrix
    }
    
    while True:
        print_menu()
        choice = input("\nSelect exercise (1-8) or 'q' to quit: ").lower()
        
        if choice == 'q':
            break
            
        try:
            if choice == '1':
                results['F'], results['points_left'], results['points_right'] = (
                    exercise1_fundamental_matrix()
                )
            elif choice == '2':
                exercise2_epipolar_geometry()
            elif choice == '3':
                exercise3_visual_verification(
                    results.get('F'),
                    results.get('points_left'),
                    results.get('points_right')
                )
            elif choice == '4':
                exercise4_mathematical_verification(
                    results.get('F'),
                    results.get('points_left'),
                    results.get('points_right')
                )
            elif choice == '5':
                exercise5_triangulation()
            elif choice == '6':
                exercise6_multiple_points()
            elif choice == '7':
                results['H'] = exercise7_homography_estimation()
            elif choice == '8':
                exercise8_homography_interactive(results.get('H'))
            else:
                print("Invalid choice. Please select 1-8 or 'q'")
                
        except Exception as e:
            print(f"\nError in exercise {choice}: {str(e)}")
            print("Please try again or select another exercise")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()