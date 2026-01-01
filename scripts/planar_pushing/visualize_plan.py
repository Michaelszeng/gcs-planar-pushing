import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from gcs_planar_pushing.geometry.planar.planar_pushing_trajectory import (
    PlanarPushingTrajectory,
)
from gcs_planar_pushing.visualize.planar_pushing import (
    visualize_planar_pushing_trajectory,
)

pkl_file = "trajectories/tee_trajectory/trajectory/traj_rounded.pkl"


def visualize_plan(debug: bool = False):
    traj = PlanarPushingTrajectory.load(pkl_file)

    # Create time samples and collect poses
    time_samples = np.arange(traj.start_time, traj.end_time, 0.5)
    print(f"traj.start_time: {traj.start_time}, traj.end_time: {traj.end_time}")

    pose_list = []
    for t in time_samples:
        pose_list.append(traj.get_pusher_planar_pose(t))
    print(f"pose_list: {pose_list}")

    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(10, 10))
    ax: Axes  # Type hint for linter

    # Extract x, y from poses
    x_coords = [pose.x for pose in pose_list]
    y_coords = [pose.y for pose in pose_list]

    # Plot x-y trajectory color-coded by time
    scatter = ax.scatter(x_coords, y_coords, c=time_samples, cmap="viridis", s=20, alpha=0.8, edgecolors="none")
    ax.plot(
        x_coords[0],
        y_coords[0],
        "go",
        markersize=12,
        label="Start",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    ax.plot(
        x_coords[-1],
        y_coords[-1],
        "ro",
        markersize=12,
        label="End",
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Slider Trajectory (X-Y)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time (s)")

    plt.tight_layout()
    plt.savefig(pkl_file.replace(".pkl", ".png"))

    # visualize_planar_pushing_trajectory(traj, show=True)


if __name__ == "__main__":
    visualize_plan(debug=True)
