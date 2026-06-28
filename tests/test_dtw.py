"""Tests for Dynamic Time Warping (DTW) metrics."""

import pytest
import torch

from robometric_frame.trajectory_quality import DTWDistance, NormalizedDTW, SuccessWeightedDTW


class TestDTWDistance:
    """Test suite for DTWDistance metric."""

    def test_identical_trajectories(self) -> None:
        """Test DTW distance for identical trajectories (should be zero)."""
        metric = DTWDistance()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0))

    def test_different_lengths_same_path(self) -> None:
        """Test DTW with trajectories of different lengths covering same path."""
        metric = DTWDistance()
        # Reference: 4 points along a straight line
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # Predicted: 7 points along the same line (different density)
        predicted = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.0],
                [2.0, 0.0],
                [2.5, 0.0],
                [3.0, 0.0],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # DTW should be small since paths are aligned (extra 0.5-step costs accumulate to 1.5)
        assert result < 2.0

    def test_temporal_shift_hesitation(self) -> None:
        """Test DTW handles temporal shift (hesitation at start)."""
        metric = DTWDistance()
        # Reference: direct path
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        # Predicted: hesitates at start, then catches up
        predicted = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # DTW should be small (tolerates hesitation)
        assert result < 1.0

    def test_completely_different_trajectories(self) -> None:
        """Test DTW for completely different trajectories."""
        metric = DTWDistance()
        # Reference: along x-axis
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        # Predicted: along y-axis, far away
        predicted = torch.tensor([[0.0, 10.0], [0.0, 11.0], [0.0, 12.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # DTW should be large
        assert result > 10.0

    def test_3d_trajectory(self) -> None:
        """Test DTW with 3D trajectories."""
        metric = DTWDistance()
        predicted = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0))

    def test_7dof_trajectory(self) -> None:
        """Test DTW with 7-DoF action space."""
        metric = DTWDistance()
        # Simulating 7-DoF robot arm action space
        predicted = torch.randn(10, 7)
        reference = predicted.clone()  # Identical
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_1d_trajectory(self) -> None:
        """Test DTW with 1D trajectories."""
        metric = DTWDistance()
        predicted = torch.tensor([[0.0], [1.0], [2.0]])
        reference = torch.tensor([[0.0], [1.0], [2.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0))

    def test_single_point_trajectories(self) -> None:
        """Test DTW with single-point trajectories."""
        metric = DTWDistance()
        predicted = torch.tensor([[1.0, 2.0]])
        reference = torch.tensor([[0.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Should be Euclidean distance between the two points
        expected = torch.sqrt(torch.tensor(1.0**2 + 2.0**2))
        assert torch.isclose(result, expected)

    def test_multiple_updates(self) -> None:
        """Test metric with multiple trajectory updates."""
        metric = DTWDistance()
        # First pair: identical (DTW=0)
        pred1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(pred1, ref1)

        # Second pair: offset by 1 unit
        pred2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref2 = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        metric.update(pred2, ref2)

        result = metric.compute()
        # Average of 0 and some positive value
        assert result > 0

    def test_reset(self) -> None:
        """Test metric reset functionality."""
        metric = DTWDistance()
        pred = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
        metric.update(pred, ref)
        result1 = metric.compute()

        metric.reset()

        # After reset, compute identical trajectories
        pred2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        ref2 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(pred2, ref2)
        result2 = metric.compute()

        assert result2 < result1  # Second should be 0, first was positive

    def test_invalid_shape_1d_raises(self) -> None:
        """Test that 1D tensor raises ValueError."""
        metric = DTWDistance()
        with pytest.raises(ValueError, match="2 dimensions"):
            metric.update(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([[0.0, 0.0]]))

    def test_mismatched_dimensionality_raises(self) -> None:
        """Test that mismatched D raises ValueError."""
        metric = DTWDistance()
        with pytest.raises(ValueError, match="same dimensionality"):
            metric.update(
                torch.tensor([[0.0, 0.0]]),  # D=2
                torch.tensor([[0.0, 0.0, 0.0]]),  # D=3
            )

    def test_empty_trajectory_raises(self) -> None:
        """Test that empty trajectory raises ValueError."""
        metric = DTWDistance()
        with pytest.raises(ValueError, match="at least 1 point"):
            metric.update(torch.tensor([]).reshape(0, 2), torch.tensor([[0.0, 0.0]]))

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = DTWDistance()
        with pytest.raises(RuntimeError, match="no trajectory pairs"):
            metric.compute()

    def test_different_dtypes(self) -> None:
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            metric = DTWDistance()
            predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
            reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
            metric.update(predicted, reference)
            result = metric.compute()
            assert torch.isclose(result, torch.tensor(0.0))

    def test_int_dtype(self) -> None:
        """Test with integer dtype (should work with automatic conversion)."""
        metric = DTWDistance()
        predicted = torch.tensor([[0, 0], [1, 0], [2, 0]])
        reference = torch.tensor([[0, 0], [1, 0], [2, 0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0))

    def test_gpu_if_available(self) -> None:
        """Test metric on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = DTWDistance().to("cuda")
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        metric.update(predicted, reference)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(0.0, device="cuda"))


class TestNormalizedDTW:
    """Test suite for NormalizedDTW metric."""

    def test_identical_trajectories(self) -> None:
        """Test nDTW for identical trajectories (should be 1.0)."""
        metric = NormalizedDTW()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0))

    def test_ndtw_range(self) -> None:
        """Test that nDTW is always between 0 and 1."""
        metric = NormalizedDTW()
        # Various trajectory pairs
        pairs = [
            (
                torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            ),
            (
                torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                torch.tensor([[0.0, 10.0], [1.0, 10.0]]),
            ),
            (
                torch.randn(5, 3),
                torch.randn(7, 3),
            ),
        ]
        for pred, ref in pairs:
            metric.reset()
            metric.update(pred, ref)
            result = metric.compute()
            assert 0.0 <= result <= 1.0, f"nDTW {result} out of range [0, 1]"

    def test_different_lengths_high_ndtw(self) -> None:
        """Test that similar paths with different lengths have high nDTW."""
        metric = NormalizedDTW()
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        predicted = torch.tensor(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.5, 0.0],
                [2.0, 0.0],
                [2.5, 0.0],
                [3.0, 0.0],
            ]
        )
        metric.update(predicted, reference)
        result = metric.compute()
        # Should be reasonably high since paths align well
        assert result > 0.6

    def test_dissimilar_trajectories_low_ndtw(self) -> None:
        """Test that dissimilar trajectories have low nDTW."""
        metric = NormalizedDTW()
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        predicted = torch.tensor([[0.0, 100.0], [1.0, 100.0], [2.0, 100.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        # Should be close to 0 for very dissimilar trajectories
        assert result < 0.1

    def test_custom_normalization_factor(self) -> None:
        """Test nDTW with custom normalization factor."""
        metric_default = NormalizedDTW()
        metric_custom = NormalizedDTW(normalization_factor=2.0)

        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])

        metric_default.update(predicted, reference)
        metric_custom.update(predicted, reference)

        result_default = metric_default.compute()
        result_custom = metric_custom.compute()

        # Results should differ due to different normalization
        assert not torch.isclose(result_default, result_custom)

    def test_invalid_normalization_factor_raises(self) -> None:
        """Test that non-positive normalization factor raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            NormalizedDTW(normalization_factor=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            NormalizedDTW(normalization_factor=-1.0)

    def test_single_point_reference(self) -> None:
        """Test nDTW with single-point reference (edge case)."""
        metric = NormalizedDTW()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0]])  # Single point
        metric.update(predicted, reference)
        result = metric.compute()
        # Should still be in valid range
        assert 0.0 <= result <= 1.0

    def test_identical_single_points(self) -> None:
        """Test nDTW with identical single-point trajectories."""
        metric = NormalizedDTW()
        predicted = torch.tensor([[1.0, 2.0]])
        reference = torch.tensor([[1.0, 2.0]])
        metric.update(predicted, reference)
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0))

    def test_stationary_trajectory(self) -> None:
        """Test nDTW with stationary (all same points) reference."""
        metric = NormalizedDTW()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])  # Stationary
        metric.update(predicted, reference)
        result = metric.compute()
        # Should be valid (d defaults to 1.0 to avoid division by zero)
        assert 0.0 <= result <= 1.0

    def test_multiple_updates(self) -> None:
        """Test nDTW with multiple updates."""
        metric = NormalizedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        # Update with identical (nDTW=1.0)
        metric.update(ref.clone(), ref)
        # Update with different
        metric.update(torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]), ref)

        result = metric.compute()
        # Average should be between 0 and 1
        assert 0.0 < result < 1.0

    def test_reset(self) -> None:
        """Test nDTW reset functionality."""
        metric = NormalizedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        # First: dissimilar
        metric.update(torch.tensor([[0.0, 10.0], [1.0, 10.0]]), ref)
        result1 = metric.compute()

        metric.reset()

        # After reset: identical
        metric.update(ref.clone(), ref)
        result2 = metric.compute()

        assert result2 > result1

    def test_gpu_if_available(self) -> None:
        """Test nDTW on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = NormalizedDTW().to("cuda")
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        metric.update(predicted, reference)
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(1.0, device="cuda"))


class TestSuccessWeightedDTW:
    """Test suite for SuccessWeightedDTW metric."""

    def test_identical_trajectories_success(self) -> None:
        """Test SDTW for identical trajectories with success (should be 1.0)."""
        metric = SuccessWeightedDTW()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference, success=torch.tensor(True))
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0))

    def test_identical_trajectories_failure(self) -> None:
        """Test SDTW for identical trajectories with failure (should be 0.0)."""
        metric = SuccessWeightedDTW()
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metric.update(predicted, reference, success=torch.tensor(False))
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0))

    def test_success_as_int(self) -> None:
        """Test SDTW with success as integer tensor."""
        metric = SuccessWeightedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(ref.clone(), ref, success=torch.tensor(1))
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0))

        metric.reset()
        metric.update(ref.clone(), ref, success=torch.tensor(0))
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(0.0))

    def test_success_as_float(self) -> None:
        """Test SDTW with success as float tensor."""
        metric = SuccessWeightedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        metric.update(ref.clone(), ref, success=torch.tensor(1.0))
        result = metric.compute()
        assert torch.isclose(result, torch.tensor(1.0))

    def test_sdtw_range(self) -> None:
        """Test that SDTW is always between 0 and 1."""
        metric = SuccessWeightedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        # Various scenarios
        scenarios = [
            (ref.clone(), torch.tensor(True)),
            (ref.clone(), torch.tensor(False)),
            (torch.randn(5, 2), torch.tensor(True)),
            (torch.randn(7, 2), torch.tensor(False)),
        ]

        for pred, success in scenarios:
            metric.reset()
            metric.update(pred, ref, success=success)
            result = metric.compute()
            assert 0.0 <= result <= 1.0

    def test_multiple_updates_mixed_success(self) -> None:
        """Test SDTW with mixed success/failure updates."""
        metric = SuccessWeightedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        # Identical + success: SDTW = 1.0
        metric.update(ref.clone(), ref, success=torch.tensor(True))
        # Identical + failure: SDTW = 0.0
        metric.update(ref.clone(), ref, success=torch.tensor(False))

        result = metric.compute()
        # Average of 1.0 and 0.0 = 0.5
        assert torch.isclose(result, torch.tensor(0.5))

    def test_custom_normalization_factor(self) -> None:
        """Test SDTW with custom normalization factor."""
        metric = SuccessWeightedDTW(normalization_factor=1.0)
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = torch.tensor([[0.0, 0.5], [1.0, 0.5], [2.0, 0.5]])
        metric.update(predicted, reference, success=torch.tensor(True))
        result = metric.compute()
        assert 0.0 <= result <= 1.0

    def test_invalid_normalization_factor_raises(self) -> None:
        """Test that non-positive normalization factor raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            SuccessWeightedDTW(normalization_factor=0.0)

    def test_reset(self) -> None:
        """Test SDTW reset functionality."""
        metric = SuccessWeightedDTW()
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

        # First: failure
        metric.update(ref.clone(), ref, success=torch.tensor(False))
        result1 = metric.compute()
        assert torch.isclose(result1, torch.tensor(0.0))

        metric.reset()

        # After reset: success
        metric.update(ref.clone(), ref, success=torch.tensor(True))
        result2 = metric.compute()
        assert torch.isclose(result2, torch.tensor(1.0))

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_compute_before_update_raises(self) -> None:
        """Test that compute before update raises RuntimeError."""
        metric = SuccessWeightedDTW()
        with pytest.raises(RuntimeError, match="no trajectory pairs"):
            metric.compute()

    def test_gpu_if_available(self) -> None:
        """Test SDTW on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metric = SuccessWeightedDTW().to("cuda")
        predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        reference = torch.tensor([[0.0, 0.0], [1.0, 0.0]], device="cuda")
        metric.update(predicted, reference, success=torch.tensor(True, device="cuda"))
        result = metric.compute()
        assert result.device.type == "cuda"
        assert torch.isclose(result, torch.tensor(1.0, device="cuda"))


class TestDTWIntegration:
    """Integration tests for DTW metrics."""

    def test_all_metrics_consistent(self) -> None:
        """Test that DTW, nDTW, and SDTW are consistent."""
        dtw = DTWDistance()
        ndtw = NormalizedDTW()
        sdtw = SuccessWeightedDTW()

        pred = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        dtw.update(pred, ref)
        ndtw.update(pred, ref)
        sdtw.update(pred, ref, success=torch.tensor(True))

        dtw_result = dtw.compute()
        ndtw_result = ndtw.compute()
        sdtw_result = sdtw.compute()

        # Identical trajectories: DTW=0, nDTW=1, SDTW=1
        assert torch.isclose(dtw_result, torch.tensor(0.0))
        assert torch.isclose(ndtw_result, torch.tensor(1.0))
        assert torch.isclose(sdtw_result, torch.tensor(1.0))

    def test_dtw_vs_mse_temporal_shift(self) -> None:
        """Test that DTW handles temporal shifts better than MSE would."""
        dtw = DTWDistance()

        # Reference trajectory
        ref = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        # Predicted: same path but with hesitation (pause at start)
        pred_hesitation = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        # Predicted: same path but faster
        pred_fast = torch.tensor([[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        dtw.update(pred_hesitation, ref)
        dtw_hesitation = dtw.compute()

        dtw.reset()
        dtw.update(pred_fast, ref)
        dtw_fast = dtw.compute()

        # Both should have relatively low DTW (good alignment)
        assert dtw_hesitation < 2.0
        assert dtw_fast < 2.0

    def test_vla_action_chunking_scenario(self) -> None:
        """Test realistic VLA model evaluation scenario with action chunking."""
        ndtw = NormalizedDTW()

        # Demonstration trajectory (what human did)
        demo = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # Start
                [0.5, 0.0, 0.0],  # Reach
                [1.0, 0.0, 0.0],  # Reach
                [1.0, 0.5, 0.0],  # Adjust
                [1.0, 1.0, 0.0],  # Grasp position
            ]
        )

        # Model A: Same path, different chunk boundaries
        model_a = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.3, 0.0],
                [1.0, 0.6, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )

        # Model B: Wrong direction
        model_b = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, -0.5, 0.0],
                [-1.0, -1.0, 0.0],
            ]
        )

        ndtw.update(model_a, demo)
        score_a = ndtw.compute()

        ndtw.reset()
        ndtw.update(model_b, demo)
        score_b = ndtw.compute()

        # Model A should score higher than Model B
        assert score_a > score_b
        assert score_a > 0.7  # Good trajectory
        assert score_b < 0.3  # Bad trajectory

    def test_large_trajectory(self) -> None:
        """Test with large trajectories (memory/performance check)."""
        dtw = DTWDistance()
        # Create trajectories with 100 points each
        pred = torch.cumsum(torch.randn(100, 3) * 0.1, dim=0)
        ref = torch.cumsum(torch.randn(100, 3) * 0.1, dim=0)

        dtw.update(pred, ref)
        result = dtw.compute()

        # Just verify it computes without error
        assert result.ndim == 0
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
