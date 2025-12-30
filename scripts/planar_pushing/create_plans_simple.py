from planning_through_contact.experiments.utils import get_default_plan_config, get_default_solver_params
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import PlanarPushingStartAndGoal
from planning_through_contact.planning.planar.utils import create_plan

solver_params = get_default_solver_params()

slider_initial_pose = PlanarPose(0.14, 0.05, -0.8)
slider_type = "sugar_box"

# Target poses
slider_target_pose = PlanarPose(0.0, 0.0, 0.0)
pusher_target_pose = PlanarPose(-0.3, 0, 0)

config = get_default_plan_config(slider_type=slider_type, use_case="normal")
start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose=slider_initial_pose,
    slider_target_pose=slider_target_pose,
    pusher_initial_pose=PlanarPose(-0.3, 0, 0),
    pusher_target_pose=pusher_target_pose,
)

print(f"Starting planning for slider type: {slider_type}")

# create_plan handles planning, folder creation, trajectory saving, and visualization
result = create_plan(
    start_and_target=start_and_goal,
    config=config,
    solver_params=solver_params,
    output_folder="trajectories",
    output_name=f"{slider_type}_trajectory",
    save_video=True,
    interpolate_video=True,
    do_rounding=True,
    save_traj=True,
    debug=True,
)

# t = 11.5

# active_vertices = retrieve_mode_sequence(result.path, t=t)
# print(f"Active vertices: {active_vertices}")

# start_and_goal = PlanarPushingStartAndGoal(
#     slider_initial_pose=result.trajectory.get_slider_planar_pose(t),
#     slider_target_pose=slider_target_pose,
#     pusher_initial_pose=result.trajectory.get_pusher_planar_pose(t),
#     pusher_target_pose=pusher_target_pose,
# )

# print("Starting planning using fixed mode sequence...")

# result = create_plan(
#     start_and_target=start_and_goal,
#     config=config,
#     solver_params=solver_params,
#     active_vertices=active_vertices,
#     output_folder="trajectories",
#     output_name=f"{slider_type}_trajectory",
#     save_video=True,
#     interpolate_video=True,
#     do_rounding=True,
#     save_traj=True,
#     debug=True,
# )

# print("Done!")
