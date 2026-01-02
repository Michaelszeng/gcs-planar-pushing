import os
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt

from gcs_planar_pushing.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from gcs_planar_pushing.geometry.planar.face_contact import FaceContactMode
from gcs_planar_pushing.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from gcs_planar_pushing.geometry.planar.non_collision_subgraph import (
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.geometry.planar.planar_pushing_path import PlanarPushingPath
from gcs_planar_pushing.geometry.planar.planar_pushing_trajectory import (
    NonCollisionTrajSegment,
)
from gcs_planar_pushing.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
)
from gcs_planar_pushing.planning.planar.planar_pushing_planner import PlanarPushingPlanner
from gcs_planar_pushing.visualize.colors import COLORS
from gcs_planar_pushing.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_trajectory,
)

GcsVertex = opt.GraphOfConvexSets.Vertex


class PlanarPushingMPC:
    def __init__(
        self,
        config: PlanarPlanConfig,
        start_and_goal: PlanarPushingStartAndGoal,
        solver_params: PlanarSolverParams,
        plan: bool = True,
    ):
        """
        Formulate initial GCS Problem and do full solve
        """
        self.config = config
        self.solver_params = solver_params
        self.planner = PlanarPushingPlanner(config)
        self.planner.config.start_and_goal = start_and_goal
        self.planner.formulate_problem()

        if plan:
            path = self.planner.plan_path(solver_params, store_result=False)
            assert path is not None
            self.original_path = path
            self.original_traj = path.to_traj()
        else:
            self.original_path = None
            self.original_traj = None

        # Store "short" mode vertex, which is a copy of the original vertex in the GCS but with a shorter time to
        # account for the passage of time during MPC iterations.
        self.previous_short_mode_vertex = None  # Optional[GcsVertex]

        # Cache the collision-free region for the source vertex to avoid expensive recomputation
        self.cached_source_region: Optional[PolytopeContactLocation] = None

    def load_original_path(self, filename: str) -> None:
        """
        Loads the original path from a file.
        """
        all_pairs = self.planner._get_all_vertex_mode_pairs()
        self.original_path = PlanarPushingPath.load(filename, self.planner.gcs, all_pairs)
        self.original_traj = self.original_path.to_traj()

    def _get_collision_free_region_at_time(self, t: float) -> PolytopeContactLocation:
        """
        Extract the collision-free region/mode that the pusher is in at time t.
        """
        # Get the current segment from the original trajectory
        if t >= self.original_traj.end_time:
            segment_idx = len(self.original_traj.traj_segments) - 1
        else:
            segment_idx = np.where(t < self.original_traj.end_times)[0][0]

        # Extract region index from the corresponding segment
        pair = self.original_path.pairs[segment_idx]
        mode = pair.mode

        # Return the collision-free region (stored in contact_location field)
        return mode.contact_location

    def _get_remaining_mode_sequence(
        self,
        plan_start: Optional[PlanarPushingStartAndGoal] = None,
        t: Optional[float] = None,
    ) -> List[str]:
        """
        Retrieves the remaining mode sequence (list of active vertices) from a PlanarPushingPath.

        Currently supports two methods of determining the mode sequence:
        - Using the plan start and searching for the best matching time and finding the starting mode at that time
        - Using t and finding the starting mode at that time
        """
        if plan_start is None and t is None:  # Return full mode sequence
            return self.original_path.get_path_names()

        traj = self.original_traj

        if t is not None:
            best_t = t
        else:
            assert plan_start is not None

            # Find the time t that minimizes the distance to the current state
            target_slider = plan_start.slider_initial_pose
            target_pusher = plan_start.pusher_initial_pose

            # Helper to compute distance
            def _dist_at_t(t_: float) -> float:
                slider_pose = traj.get_slider_planar_pose(t_)
                pusher_pose = traj.get_pusher_planar_pose(t_)

                # Position error
                pos_err = np.linalg.norm(slider_pose.pos() - target_slider.pos())

                # Angle error (handle wrapping)
                th_err = np.abs(slider_pose.theta - target_slider.theta)
                th_err = min(th_err, 2 * np.pi - th_err)

                # Pusher error
                if target_pusher is not None:
                    pusher_err = np.linalg.norm(pusher_pose.pos() - target_pusher.pos())
                else:
                    pusher_err = 0.0

                return pos_err + 0.5 * th_err + pusher_err

            # Coarse search over the trajectory
            # We use a relatively fine resolution to ensure we catch the right segment
            NUM_STEPS = 100
            ts = np.linspace(traj.start_time, traj.end_time, NUM_STEPS)
            dists = [_dist_at_t(t) for t in ts]
            best_idx = np.argmin(dists)
            best_t = ts[best_idx]

        # Identify the segment index using the trajectory logic
        # We use the internal method _get_curr_segment_idx if available, or replicate the logic
        if hasattr(traj, "_get_curr_segment_idx"):
            segment_idx = traj._get_curr_segment_idx(best_t)  # type: ignore
        else:
            # Replicate logic: idx = first index where t < end_time
            # or last index if t >= all end_times
            if best_t >= traj.end_time:
                segment_idx = len(traj.traj_segments) - 1
            else:
                segment_idx = np.where(best_t < traj.end_times)[0][0]

        mode_sequence = [pair.vertex.name() for pair in self.original_path.pairs[segment_idx:]]

        # We must always start with the source mode for the planner to work
        if mode_sequence[0] != "source":
            mode_sequence.insert(0, "source")

        return mode_sequence

    def _update_initial_poses(
        self,
        t: float,
        current_slider_pose: PlanarPose,
        current_pusher_pose: PlanarPose,
        mode_sequence: Optional[List[str]] = None,
        current_pusher_velocity: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        # Update config
        assert self.planner.config.start_and_goal is not None
        self.planner.config.start_and_goal.slider_initial_pose = current_slider_pose
        self.planner.config.start_and_goal.pusher_initial_pose = current_pusher_pose

        # Clean up old graph elements to prevent bloat
        # 1. Remove old source vertex
        old_source_pair = self.planner.source
        if old_source_pair is not None:
            old_vertex = old_source_pair.vertex
            old_name = old_vertex.name()

            # Remove from GCS
            self.planner.gcs.RemoveVertex(old_vertex)

            # Remove stale edges from planner.edges dictionary
            # Keys are (u_name, v_name)
            keys_to_remove = [k for k in self.planner.edges.keys() if k[0] == old_name]
            for k in keys_to_remove:
                del self.planner.edges[k]

        # 2. Remove old "short" mode vertex from previous iteration
        if self.previous_short_mode_vertex is not None:
            old_short_name = self.previous_short_mode_vertex.name()
            self.planner.gcs.RemoveVertex(self.previous_short_mode_vertex)
            # Remove stale edges connected to short mode from dictionary
            keys_to_remove = [k for k in self.planner.edges.keys() if k[0] == old_short_name or k[1] == old_short_name]
            for k in keys_to_remove:
                del self.planner.edges[k]
            # Remove from planner extra vertex mode pairs from previous MPC iteration
            if old_short_name in self.planner.extra_vertex_mode_pairs:
                del self.planner.extra_vertex_mode_pairs[old_short_name]
            self.previous_short_mode_vertex = None

        # Update planner source
        # This creates a new source vertex and connects it to the existing GCS instance
        # First, we need to find the region/mode that the new source is in.
        self.cached_source_region = self._get_collision_free_region_at_time(t)
        # Then, call _set_initial_poses to set this new source vertex in the GCS
        self.planner._set_initial_poses(
            current_pusher_pose, current_slider_pose, collision_free_region=self.cached_source_region
        )

        if mode_sequence is not None and len(mode_sequence) >= 2:
            first_mode_name = mode_sequence[1]

            all_pairs = self.planner._get_all_vertex_mode_pairs()

            # Check if we need to replace the first mode with a custom timed one
            if self.config.time_first_mode is not None:
                if first_mode_name in all_pairs:
                    original_pair = all_pairs[first_mode_name]
                    original_mode = original_pair.mode

                    # Create new mode with custom time
                    new_mode = None
                    if isinstance(original_mode, FaceContactMode):
                        new_mode = FaceContactMode(
                            f"{first_mode_name}_SHORT",
                            self.config.num_knot_points_contact,
                            self.config.time_first_mode,
                            original_mode.contact_location,
                            self.config,
                        )
                    elif isinstance(original_mode, NonCollisionMode):
                        new_mode = NonCollisionMode(
                            f"{first_mode_name}_SHORT",
                            self.config.num_knot_points_non_collision,
                            self.config.time_first_mode,
                            original_mode.contact_location,
                            self.config,
                        )

                    if new_mode is not None:
                        # Enforce initial velocity constraint if applicable
                        if isinstance(new_mode, NonCollisionMode) and current_pusher_velocity is not None:
                            # Constrain initial velocity
                            # v_world = v_pusher_velocity  (pusher velocity in world frame)
                            # v_body = R_WB.T @ v_world  (pusher velocity in slider body frame)
                            # (slider is static during non-collision mode)
                            R_WB = current_slider_pose.two_d_rot_matrix()
                            v_body = R_WB.T @ current_pusher_velocity

                            # Velocity constraint for Bezier curve:
                            # For a Bezier curve of degree n parameterized over [0, T]:
                            #   dB/dt |_{t=0} = n/T * (P1 - P0)
                            # So to achieve initial velocity v_body:
                            #   v_body = order / time_in_mode * (c1 - c0)
                            #   c1 - c0 = v_body * time_in_mode / order
                            decision_vars = new_mode.variables
                            assert isinstance(decision_vars, NonCollisionVariables)
                            c0_x = decision_vars.p_BP_xs[0]  # 1st control point x
                            c0_y = decision_vars.p_BP_ys[0]  # 1st control point y
                            c1_x = decision_vars.p_BP_xs[1]  # 2nd control point x
                            c1_y = decision_vars.p_BP_ys[1]  # 2nd control point y

                            order = decision_vars.num_knot_points - 1
                            scale = decision_vars.time_in_mode / order
                            new_mode.prog.AddLinearEqualityConstraint(c1_x - c0_x == v_body[0] * scale)
                            new_mode.prog.AddLinearEqualityConstraint(c1_y - c0_y == v_body[1] * scale)

                        # Add to graph
                        new_vertex = self.planner.gcs.AddVertex(new_mode.get_convex_set(), new_mode.name)
                        new_mode.add_cost_to_vertex(new_vertex)
                        if isinstance(new_mode, NonCollisionMode):
                            new_mode.add_constraints_to_vertex(new_vertex)

                        new_pair = VertexModePair(new_vertex, new_mode)
                        self.previous_short_mode_vertex = new_vertex
                        # Register new short mode vertex in planner
                        self.planner.extra_vertex_mode_pairs[new_mode.name] = new_pair

                        # Connect Source -> New Mode
                        assert self.planner.source is not None
                        source_name = self.planner.source.vertex.name()
                        edge_key_source = (source_name, new_mode.name)
                        self.planner.edges[edge_key_source] = gcs_add_edge_with_continuity(
                            self.planner.gcs,
                            self.planner.source,
                            new_pair,
                            only_continuity_on_slider=False,
                            continuity_on_pusher_velocities=False,  # Source has no velocity
                        )

                        # Connect New Mode -> Next Mode
                        if len(mode_sequence) >= 3:
                            next_mode_name = mode_sequence[2]
                            if next_mode_name in all_pairs:
                                next_pair = all_pairs[next_mode_name]
                                edge_key_next = (new_mode.name, next_mode_name)

                                # Only enforce velocity continuity if both modes are NonCollisionMode
                                enforce_velocity = (
                                    self.config.continuity_on_pusher_velocity
                                    and isinstance(new_mode, NonCollisionMode)
                                    and isinstance(next_pair.mode, NonCollisionMode)
                                )

                                self.planner.edges[edge_key_next] = gcs_add_edge_with_continuity(
                                    self.planner.gcs,
                                    new_pair,
                                    next_pair,
                                    only_continuity_on_slider=False,
                                    continuity_on_pusher_velocities=enforce_velocity,
                                )

                        # Update mode sequence in place so planner uses the new mode
                        mode_sequence[1] = new_mode.name
                        return

    def _update_mode_timing(self, t: float) -> None:
        traj = self.original_traj

        # Find current segment
        # If t > end_time, we are done / last segment
        if t >= traj.end_time:
            self.config.time_first_mode = 0.0
            return

        idx = np.where(t < traj.end_times)[0][0]
        end_time = traj.end_times[idx]
        remaining = end_time - t

        # Ensure a small positive minimum to avoid numerical issues
        self.config.time_first_mode = max(remaining, 1e-4)

    def plan(
        self,
        t: float,
        current_slider_pose: Optional[PlanarPose] = None,
        current_pusher_pose: Optional[PlanarPose] = None,
        current_pusher_velocity: Optional[npt.NDArray[np.float64]] = None,
        output_folder: str = "",
        output_name: str = "",
        save_video: bool = False,
        save_traj: bool = False,
        interpolate_video: bool = False,
        overlay_traj: bool = False,
        animation_lims: Optional[Tuple[float, float, float, float]] = None,
        hardware: bool = False,
    ) -> Optional[PlanarPushingPath]:
        """
        Using current time, plan a new path from the current state and pusher velocity.
        1) Determine remaining mode sequence from the original path
        2) Set new initial state, and update source vertex (as well as edges from the source vertex) in the GCS instance
        3) Determine the remaining time for the current mode
        4) Update self.planner.config so the first mode takes the remaining time
        5) Solve the GCS convex restriction
        6) Optionally save trajectory and video outputs
        """
        assert t > 0, f"MPC plan must be run with t > 0, current t: {t}"

        mode_sequence = self._get_remaining_mode_sequence(t=t)

        # If not provided, get from trajectory (for testing purposes -- in general, user should provide)
        if current_slider_pose is None or current_pusher_pose is None or current_pusher_velocity is None:
            traj = self.original_traj
            if current_slider_pose is None:
                current_slider_pose = traj.get_slider_planar_pose(t)
            if current_pusher_pose is None:
                current_pusher_pose = traj.get_pusher_planar_pose(t)
            if current_pusher_velocity is None:
                # Extract velocity from trajectory
                seg = traj.get_traj_segment_for_time(t)
                if isinstance(seg, NonCollisionTrajSegment):
                    v_body = seg.p_BP.traj.EvalDerivative(t)
                    R_WB = seg.get_R_WB(t)[:2, :2]
                    v_world = R_WB @ v_body
                    current_pusher_velocity = v_world.flatten()

        self._update_mode_timing(t)
        self._update_initial_poses(t, current_slider_pose, current_pusher_pose, mode_sequence, current_pusher_velocity)

        path = self.planner.plan_path(self.solver_params, active_vertices=mode_sequence, store_result=False)

        # Save outputs if requested
        if path is not None and (save_video or save_traj) and output_folder:
            folder_name = f"{output_folder}/{output_name}" if output_name else output_folder
            os.makedirs(folder_name, exist_ok=True)

            traj = path.to_traj(
                rounded=False
            )  # Do not round; with fixed mode sequence, soln. is already feasible and optimal

            if save_traj:
                trajectory_folder = f"{folder_name}/trajectory"
                os.makedirs(trajectory_folder, exist_ok=True)

                if traj is not None:
                    traj.save(f"{trajectory_folder}/traj.pkl")

                slider_color = COLORS["aquamarine4"].diffuse()

                if traj is not None:
                    make_traj_figure(
                        traj,
                        filename=f"{trajectory_folder}/traj",
                        slider_color=slider_color,
                        split_on_mode_type=True,
                        show_workspace=hardware,
                    )

            if save_video:
                if traj is not None:
                    # Prepare overlay trajectories if requested
                    overlay_trajs_arg = None
                    if overlay_traj:
                        # Original trajectory: black for both slider and pusher
                        original_slider_color = COLORS["black"]
                        original_pusher_color = COLORS["black"]
                        # New trajectory: use object colors (slider=aquamarine, pusher=firebrick)
                        new_slider_color = COLORS["aquamarine4"]
                        new_pusher_color = COLORS["firebrick3"]
                        overlay_trajs_arg = [
                            (self.original_traj, original_slider_color, original_pusher_color),
                            (traj, new_slider_color, new_pusher_color),
                        ]

                    visualize_planar_pushing_trajectory(
                        traj,
                        save=True,
                        filename=f"{folder_name}/traj",
                        visualize_knot_points=not interpolate_video,
                        lims=animation_lims,
                        overlay_trajs=overlay_trajs_arg,
                    )

        return path
