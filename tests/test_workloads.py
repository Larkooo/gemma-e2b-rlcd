from scripts.benchmark_workloads import agreement, quality, summarize


def test_workload_quality_counts_missing_nested_answers_as_failures():
    expected = {"facts": {"cat": True, "dog": False}, "category": ["titanium_a", "titanium_b"]}
    actual = {"facts": {"cat": True}, "category": "titanium_b"}
    assert quality(actual, expected) == {
        "correct": 2,
        "scored": 3,
        "checks": {"facts.cat": True, "facts.dog": False, "category": True},
    }
    assert quality(None, expected)["correct"] == 0
    assert agreement(actual, None) is None
    assert agreement({"a": True}, {"a": True, "b": False}) == {"a": True, "b": False}


def test_benchmark_preserves_failed_runs_and_does_not_award_them_a_speedup():
    samples = [
        {
            "method": "batched",
            "seconds": 1,
            "valid": True,
            "values": {"a": True},
            "quality": {"correct": 1, "scored": 1},
        },
        {
            "method": "batched",
            "seconds": 3,
            "valid": True,
            "values": {"a": True},
            "quality": {"correct": 1, "scored": 1},
        },
        {
            "method": "normal",
            "seconds": 4,
            "valid": True,
            "values": {"a": True},
            "quality": {"correct": 1, "scored": 1},
        },
        {
            "method": "normal",
            "seconds": 8,
            "valid": False,
            "values": None,
            "quality": {"correct": 0, "scored": 1},
        },
    ]
    summary = summarize(samples, ["batched", "normal"])
    assert summary["batched"]["median_seconds"] == 2
    assert summary["normal"]["median_seconds"] == 6
    assert summary["normal"]["valid_runs"] == 1
    assert summary["normal"]["attempted_runs"] == 2
    assert summary["normal"]["correct"] == 1
    assert summary["normal"]["scored"] == 2
    assert summary["normal_over_batched"] is None
