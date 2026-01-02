import os
import time

from gcs_planar_pushing.experiments.utils import get_default_plan_config, get_default_solver_params
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.planning.planar.mpc import PlanarPushingMPC
from gcs_planar_pushing.planning.planar.planar_plan_config import PlanarPushingStartAndGoal

solver_params = get_default_solver_params()

slider_initial_pose = PlanarPose(0.14, 0.05, -0.8)
slider_type = "arbitrary"
arbitrary_shape_pickle_path = "arbitrary_shape_pickles/small_t_pusher.pkl"

# Target poses
slider_target_pose = PlanarPose(0.0, 0.0, 0.0)
pusher_target_pose = PlanarPose(-0.3, 0, 0)

config = get_default_plan_config(
    slider_type=slider_type, arbitrary_shape_pickle_path=arbitrary_shape_pickle_path, use_case="normal"
)
start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose=slider_initial_pose,
    slider_target_pose=slider_target_pose,
    pusher_initial_pose=PlanarPose(-0.3, 0, 0),
    pusher_target_pose=pusher_target_pose,
)

print("Constructing MPC Planner...")

# Load original plan from cache if it exists to speed up test
CACHE_PATH = "mpc_path_cache.pkl"
if os.path.exists(CACHE_PATH):
    print(f"Loading cached path from {CACHE_PATH}")
    mpc = PlanarPushingMPC(
        config,
        start_and_goal,
        solver_params,
        plan=False,
    )
    print("about to load")
    mpc.load_original_path(CACHE_PATH)
    print("finished loading")
else:
    print("Computing fresh path and caching it...")
    mpc = PlanarPushingMPC(
        config,
        start_and_goal,
        solver_params,
        plan=True,
    )
    mpc.original_path.save(CACHE_PATH)

print("Planning with MPC...")
t = 1
current_slider_pose = mpc.original_traj.get_slider_planar_pose(t)
current_pusher_pose = mpc.original_traj.get_pusher_planar_pose(t)
current_pusher_velocity = mpc.original_traj.get_pusher_velocity(t)
start = time.time()
path = mpc.plan(
    t=t,
    current_slider_pose=current_slider_pose,
    current_pusher_pose=current_pusher_pose,
    current_pusher_velocity=current_pusher_velocity,
    # output_folder="trajectories_mpc",
    # output_name=f"arbitrary_small_t_pusher_trajectory_t={t}",
    # save_video=True,
    # interpolate_video=True,
    # overlay_traj=True,
    # save_traj=True,
)
print(f"Time taken for MPC replan: {time.time() - start}")
print("Done!")
