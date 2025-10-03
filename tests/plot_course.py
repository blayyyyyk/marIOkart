import os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.course import read_collision_data, read_checkpoint_data

def plot_course(highlight_points = None):
    collision_points = read_collision_data()
    checkpoint_points = read_checkpoint_data()

    collision_x_coords = [p[0] for p in collision_points]
    collision_z_coords = [p[2] for p in collision_points]

    checkpoint_x_coords_0 = [item["p0"][0] for item in checkpoint_points]
    checkpoint_z_coords_0 = [item["p0"][1] for item in checkpoint_points]
    checkpoint_x_coords_1 = [item["p1"][0] for item in checkpoint_points]
    checkpoint_z_coords_1 = [item["p1"][1] for item in checkpoint_points]

    # Create plot
    plt.figure(figsize=(6, 6))
    # plt.scatter(collision_x_coords, collision_z_coords, color='b', marker='o')
    plt.scatter(checkpoint_x_coords_0, checkpoint_z_coords_0, color="r", marker="o")
    plt.scatter(checkpoint_x_coords_1, checkpoint_z_coords_1, color="g", marker="o")
    
    plt.scatter(collision_x_coords, collision_z_coords, color="b", marker="o")
    
    # Add labels for each point
    for x, y in zip(checkpoint_x_coords_0, checkpoint_z_coords_0):
        plt.text(x + 0.1, y + 0.1, f"({x:.1f}, {y:.1f})", fontsize=9, color="darkred")
        
    # Add labels for each point
    for x, y in zip(checkpoint_x_coords_1, checkpoint_z_coords_1):
        plt.text(x + 0.1, y + 0.1, f"({x:.1f}, {y:.1f})", fontsize=9, color="darkgreen")
    
    if highlight_points:
        highlight_points_x = [point[0] for point in highlight_points]
        highlight_points_z = [point[1] for point in highlight_points]
        plt.scatter(highlight_points_x, highlight_points_z, color="b", marker="o")

    # Add labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Plot of Points")
    plt.grid(True)

    # Show plot
    plt.show()

if __name__ == "__main__":
    plot_course()
