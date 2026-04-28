from robotic_arm_trash.cli import get_project_summary


def test_project_summary_mentions_demo_locations() -> None:
    summary = get_project_summary()
    assert "robotic-arm-trash" in summary
    assert "demos" in summary
    assert "videos" in summary
