"""Trajectory quality metrics for robotics policy evaluation.

This module provides metrics for evaluating the quality of robot trajectories,
including path length, smoothness, curvature change, trajectory errors, and
dynamic time warping metrics for temporally misaligned trajectories.
"""

from robometric_frame.trajectory_quality.absolute_trajectory_error import AbsoluteTrajectoryError
from robometric_frame.trajectory_quality.curvature_change import CurvatureChange
from robometric_frame.trajectory_quality.dtw import (
    DTWBase,
    DTWDistance,
    NormalizedDTW,
    SuccessWeightedDTW,
)
from robometric_frame.trajectory_quality.path_length import PathLength
from robometric_frame.trajectory_quality.path_smoothness import PathSmoothness
from robometric_frame.trajectory_quality.relative_trajectory_error import RelativeTrajectoryError

__all__ = [
    "AbsoluteTrajectoryError",
    "CurvatureChange",
    "DTWBase",
    "DTWDistance",
    "NormalizedDTW",
    "PathLength",
    "PathSmoothness",
    "RelativeTrajectoryError",
    "SuccessWeightedDTW",
]
