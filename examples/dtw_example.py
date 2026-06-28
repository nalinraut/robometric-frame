"""Example demonstrating DTW metrics for trajectory evaluation.

This example shows how to use DTWDistance, NormalizedDTW, and SuccessWeightedDTW
for evaluating trajectories that may have different lengths or temporal alignment.
"""

import torch

from robometric_frame import DTWDistance, NormalizedDTW, SuccessWeightedDTW


def main() -> None:
    """Demonstrate DTW metrics usage."""
    print("=" * 70)
    print("FRAME - Dynamic Time Warping (DTW) Metrics Example")
    print("=" * 70)

    # Example 1: Identical trajectories
    print("\nExample 1: Identical Trajectories")
    print("-" * 70)
    dtw = DTWDistance()
    ndtw = NormalizedDTW()
    sdtw = SuccessWeightedDTW()

    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    predicted = reference.clone()

    dtw.update(predicted, reference)
    ndtw.update(predicted, reference)
    sdtw.update(predicted, reference, success=torch.tensor(True))

    print(f"Reference trajectory: {reference.shape[0]} points")
    print(f"Predicted trajectory: {predicted.shape[0]} points (identical)")
    print(f"DTW Distance: {dtw.compute():.4f} (lower = better)")
    print(f"Normalized DTW: {ndtw.compute():.4f} (higher = better, range [0,1])")
    print(f"Success-weighted DTW: {sdtw.compute():.4f}")

    # Example 2: Different lengths (core use case)
    print("\nExample 2: Trajectories of Different Lengths")
    print("-" * 70)
    dtw.reset()
    ndtw.reset()

    # Reference: 4 points along a straight line
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # Predicted: 7 points along the same line (different density - e.g., action chunking)
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

    dtw.update(predicted, reference)
    ndtw.update(predicted, reference)

    print(f"Reference: {reference.shape[0]} points")
    print(f"Predicted: {predicted.shape[0]} points (same path, higher density)")
    print(f"DTW Distance: {dtw.compute():.4f}")
    print(f"Normalized DTW: {ndtw.compute():.4f}")
    print("Note: High nDTW because the paths align well despite different lengths")

    # Example 3: Temporal shift (hesitation)
    print("\nExample 3: Temporal Shift (Hesitation at Start)")
    print("-" * 70)
    dtw.reset()
    ndtw.reset()

    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    # Predicted: hesitates at start, then catches up
    predicted = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    dtw.update(predicted, reference)
    ndtw.update(predicted, reference)

    print(f"Reference: {reference.tolist()}")
    print(f"Predicted: {predicted.tolist()}")
    print(f"DTW Distance: {dtw.compute():.4f}")
    print(f"Normalized DTW: {ndtw.compute():.4f}")
    print("Note: DTW tolerates hesitation - MSE would heavily penalize this!")

    # Example 4: Success-weighted DTW
    print("\nExample 4: Success-weighted DTW")
    print("-" * 70)
    sdtw = SuccessWeightedDTW()

    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    predicted = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    # Success case
    sdtw.update(predicted, reference, success=torch.tensor(True))
    print(f"Task succeeded: SDTW = {sdtw.compute():.4f}")

    # Failure case
    sdtw.reset()
    sdtw.update(predicted, reference, success=torch.tensor(False))
    print(f"Task failed: SDTW = {sdtw.compute():.4f}")
    print("Note: SDTW=0 when task fails, regardless of trajectory quality")

    # Example 5: Comparing models
    print("\nExample 5: Comparing Different Models")
    print("-" * 70)

    # Demonstration trajectory (what human did)
    demo = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Start
            [0.5, 0.0, 0.0],  # Reach phase
            [1.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],  # Adjust phase
            [1.0, 1.0, 0.0],  # Target position
        ]
    )

    # Model A: Good trajectory (follows demonstration)
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

    # Model B: Poor trajectory (goes wrong direction)
    model_b = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, -0.5, 0.0],
            [-1.0, -1.0, 0.0],
        ]
    )

    ndtw_a = NormalizedDTW()
    ndtw_b = NormalizedDTW()

    ndtw_a.update(model_a, demo)
    ndtw_b.update(model_b, demo)

    print("Demonstration trajectory (3D end-effector position):")
    print(f"  Points: {demo.shape[0]}")
    print("\nModel A (follows demonstration, different timing):")
    print(f"  Points: {model_a.shape[0]}")
    print(f"  nDTW: {ndtw_a.compute():.4f}")
    print("\nModel B (wrong direction):")
    print(f"  Points: {model_b.shape[0]}")
    print(f"  nDTW: {ndtw_b.compute():.4f}")

    # Example 6: 7-DoF robotic arm
    print("\nExample 6: 7-DoF Robotic Arm Evaluation")
    print("-" * 70)
    ndtw = NormalizedDTW()

    # Simulated 7-DoF joint trajectory (40 timesteps)
    torch.manual_seed(42)
    demo_7dof = torch.cumsum(torch.randn(40, 7) * 0.1, dim=0)
    # Model prediction (47 timesteps - hesitated during execution)
    pred_7dof = torch.cumsum(torch.randn(47, 7) * 0.1, dim=0)
    # Make it similar to demo by adding demo values
    pred_7dof[:40] = demo_7dof + torch.randn(40, 7) * 0.05  # Add noise

    ndtw.update(pred_7dof, demo_7dof)
    print(f"Demo trajectory: {demo_7dof.shape} (timesteps, DoF)")
    print(f"Predicted: {pred_7dof.shape} (different length)")
    print(f"nDTW Score: {ndtw.compute():.4f}")

    # Example 7: Multiple trajectory evaluation
    print("\nExample 7: Evaluating Multiple Episodes")
    print("-" * 70)
    sdtw = SuccessWeightedDTW()

    # Simulate 5 episodes with varying success
    torch.manual_seed(123)
    episodes = [
        (torch.randn(10, 3), torch.randn(10, 3), True),
        (torch.randn(12, 3), torch.randn(10, 3), True),
        (torch.randn(8, 3), torch.randn(10, 3), False),
        (torch.randn(10, 3), torch.randn(10, 3), True),
        (torch.randn(11, 3), torch.randn(10, 3), False),
    ]

    for i, (pred, ref, success) in enumerate(episodes):
        sdtw.update(pred, ref, success=torch.tensor(success))
        print(
            f"  Episode {i + 1}: pred={pred.shape[0]}pts, ref={ref.shape[0]}pts, success={success}"
        )

    print(f"\nAverage SDTW across all episodes: {sdtw.compute():.4f}")
    print("(This combines trajectory quality with task success)")

    # Example 8: Custom normalization factor
    print("\nExample 8: Custom Normalization Factor")
    print("-" * 70)

    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    predicted = torch.tensor([[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]])  # Small offset

    ndtw_auto = NormalizedDTW()
    ndtw_custom = NormalizedDTW(normalization_factor=0.5)

    ndtw_auto.update(predicted, reference)
    ndtw_custom.update(predicted, reference)

    print(f"Automatic normalization: nDTW = {ndtw_auto.compute():.4f}")
    print(f"Custom normalization (d=0.5): nDTW = {ndtw_custom.compute():.4f}")
    print("Note: Custom d allows tuning sensitivity to trajectory differences")

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("-" * 70)
    print("1. DTW Distance: Raw alignment cost (lower = more similar)")
    print("2. nDTW: Normalized to [0,1] (higher = more similar)")
    print("3. SDTW: nDTW weighted by task success")
    print("")
    print("When to use DTW over MSE/ATE:")
    print("  - Trajectories have different lengths")
    print("  - Temporal alignment varies (hesitation, speed differences)")
    print("  - Using action chunking (ACT, Diffusion Policy)")
    print("  - Comparing across demonstrations with different timing")
    print("=" * 70)


if __name__ == "__main__":
    main()
