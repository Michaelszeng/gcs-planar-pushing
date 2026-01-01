import os
import pickle
import sys

import numpy as np

# Add repo root to path
sys.path.append(os.getcwd())

from gcs_planar_pushing.geometry.planar.planar_pushing_trajectory import PlanarPushingTrajectory

path = "trajectories_mpc/sugar_box_trajectory_t=11.5/trajectory/traj.pkl"

try:
    traj = PlanarPushingTrajectory.load(path)

    start_t = traj.start_time
    # First mode duration
    first_mode_duration = traj.traj_segments[0].end_time - traj.traj_segments[0].start_time
    print(f"First mode duration: {first_mode_duration}")

    end_first_mode_t = start_t + first_mode_duration

    print(f"Pusher at start: {traj.get_pusher_planar_pose(start_t)}")
    print(f"Pusher at end of first mode ({end_first_mode_t}): {traj.get_pusher_planar_pose(end_first_mode_t)}")

    # Check if there is a jump to second mode
    start_second_mode_t = traj.traj_segments[1].start_time
    print(f"Second mode start time: {start_second_mode_t}")
    print(f"Pusher at start of second mode: {traj.get_pusher_planar_pose(start_second_mode_t)}")

except Exception as e:
    print(f"Error loading or inspecting trajectory: {e}")
    import traceback

    traceback.print_exc()
