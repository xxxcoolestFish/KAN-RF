import torch

from cpbn.hopper_source_twin import JointStateSupportCalibrator


def test_joint_support_detects_unseen_coordinate_combination():
    coordinate = torch.linspace(-1.0, 1.0, 101)
    reference = torch.stack((coordinate, coordinate), dim=-1)
    error = torch.full((reference.shape[0],), 0.1)
    calibrator = JointStateSupportCalibrator(
        reference,
        error,
        torch.ones(2),
        neighbors=5,
    )

    familiar = calibrator.score(torch.tensor([[0.1, 0.1]]))
    joint_ood = calibrator.score(torch.tensor([[0.8, -0.8]]))

    assert joint_ood["coverage_ratio"].item() > (
        5.0 * familiar["coverage_ratio"].item()
    )
    assert joint_ood["confidence"].item() < familiar["confidence"].item()


def test_joint_support_rejects_invalid_reference_count():
    try:
        JointStateSupportCalibrator(
            torch.zeros(4, 2),
            torch.zeros(4),
            torch.ones(2),
            neighbors=4,
        )
    except ValueError as error:
        assert "more reference states" in str(error)
    else:
        raise AssertionError("expected invalid neighbor count to fail")
