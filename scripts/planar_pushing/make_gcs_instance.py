from pathlib import Path
from gcs_planar_pushing.experiments.utils import get_default_experiment_plans
from gcs_planar_pushing.experiments.utils import (
    get_default_plan_config,
    get_default_solver_params,
)
from gcs_planar_pushing.planning.planar.planar_pushing_planner import (
    PlanarPushingPlanner,
)
from gcs_planar_pushing.visualize.planar_pushing import (
    compare_trajs,
    make_traj_figure,
    plot_forces,
    visualize_planar_pushing_start_and_goal,
    visualize_planar_pushing_trajectory,
)
from gcs_planar_pushing.geometry.planar.planar_pushing_path import (
    PlanarPushingPath,
)
from pydrake.geometry.optimization import GraphOfConvexSets


def save_relaxed_and_rounded_path(path: PlanarPushingPath, output_folder: Path) -> None:
    traj_relaxed = path.to_traj()
    traj_rounded = path.to_traj(rounded=True)

    make_traj_figure(
        traj_relaxed,
        filename=f"{output_folder}/relaxed_traj",
        split_on_mode_type=True,
        show_workspace=False,
    )

    if traj_rounded is not None:
        make_traj_figure(
            traj_rounded,
            filename=f"{output_folder}/rounded_traj",
            split_on_mode_type=True,
            show_workspace=False,
        )

        compare_trajs(
            traj_relaxed,
            traj_rounded,
            traj_a_legend="relaxed",
            traj_b_legend="rounded",
            filename=f"{output_folder}/comparison",
        )

        visualize_planar_pushing_trajectory(
            traj_relaxed,  # type: ignore
            save=True,
            filename=f"{output_folder}/relaxed_traj",
            visualize_knot_points=False,
        )

        visualize_planar_pushing_trajectory(
            traj_rounded,  # type: ignore
            save=True,
            filename=f"{output_folder}/rounded_traj",
            visualize_knot_points=False,
        )


def create_gcs_instance(solve_problem: bool = True) -> GraphOfConvexSets:
    slider_type = "sugar_box"
    pusher_radius = 0.015

    config = get_default_plan_config(
        slider_type=slider_type,
        pusher_radius=pusher_radius,
    )

    seed = 1
    num_boundary_conditions = 50
    idx_to_plan_for = 2  # change this if you want to try different boundary conditions
    start_and_target = get_default_experiment_plans(
        seed, num_boundary_conditions, config
    )[idx_to_plan_for]

    output_dir = Path("OUTPUTS")
    output_dir.mkdir(exist_ok=True)

    DEBUG = True
    if DEBUG:
        visualize_planar_pushing_start_and_goal(
            config.dynamics_config.slider.geometry,
            config.dynamics_config.pusher_radius,
            start_and_target,
            save=True,
            filename=f"{output_dir}/start_and_goal",
        )

    planner = PlanarPushingPlanner(config)
    planner.config.start_and_goal = start_and_target
    planner.formulate_problem()

    gcs_instance = planner.gcs

    if solve_problem:
        solver_params = get_default_solver_params(debug=DEBUG)
        path = planner.plan_path(solver_params)

        # We may get infeasible
        if path is not None:
            save_relaxed_and_rounded_path(path, output_dir)

    return gcs_instance


if __name__ == "__main__":
    gcs = create_gcs_instance()
