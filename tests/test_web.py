import io
import json
import shutil
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from gemma_rlcd.comparison import generation_task
from gemma_rlcd.core import TokenScores, parse_question
from gemma_rlcd.web import create_app, read_spec


class FakeBackend:
    def __init__(self):
        self.last_stats = {"branch_batch_sizes": [1], "prefix_prefills": 1}
        self.calls = []

    def symbols(self, count):
        return tuple(chr(65 + i) for i in range(count))

    def score_batch(self, state, requests):
        self.calls.append((state, requests))
        for path in (*state.images, *state.audio, *state.videos):
            from pathlib import Path

            assert Path(path).is_file()
        return [
            TokenScores(tuple(float(-i) for i in range(len(request.symbols))), 1.0, 123)
            for request in requests
        ]


@pytest.fixture
def playground(tmp_path):
    backend = FakeBackend()
    app = create_app("test-model", tmp_path, backend_factory=lambda: backend)
    with TestClient(app, base_url="http://localhost") as client:
        for _ in range(100):
            if client.get("/api/status").json()["ready"]:
                break
            time.sleep(0.01)
        assert client.get("/api/status").json()["ready"]
        yield client, backend, tmp_path


def request_spec():
    return {
        "text": "A cat",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal?",
                "criteria": {"cat": "A cat", "dog": "A dog"},
            }
        },
    }


def test_large_demo_presets_are_served_and_all_decisions_reach_backend(playground):
    client, backend, _ = playground
    response = client.get("/static/demo-presets.json")
    assert response.status_code == 200
    presets = response.json()
    for name, count in [
        ("support_matrix", 28),
        ("inbox_matrix", 32),
        ("ticket_flags", 32),
        ("catalog_choices", 4),
    ]:
        spec = presets[name]
        response = client.post("/api/run", data={"spec": json.dumps(spec)})
        assert response.status_code == 200
        assert len(backend.calls[-1][1]) == count
        assert backend.calls[-1][0].text == spec["text"]
        assert set(response.json()["answers"]) == set(spec["questions"])


def test_demo_presets_preserve_the_benchmark_prompt_after_editor_serialization(playground):
    client, _, _ = playground
    presets = client.get("/static/demo-presets.json").json()
    cases = json.loads((Path(__file__).parents[1] / "examples/demo-workloads.json").read_text())
    cases = {case["name"]: case for case in cases["cases"]}
    for preset_name, case_name in [
        ("support_matrix", "support_28"),
        ("inbox_matrix", "inbox_32"),
        ("ticket_flags", "ticket_flags_32"),
        ("catalog_choices", "catalog_64"),
    ]:
        preset = presets[preset_name]
        # The editor materializes default yes/no meanings into an explicit map.
        assert all("criteria" in question for question in preset["questions"].values())
        _, actual = read_spec(json.dumps(preset))
        expected = {
            name: parse_question(raw) for name, raw in cases[case_name]["questions"].items()
        }
        assert preset["text"] == cases[case_name]["text"]
        assert generation_task(actual) == generation_task(expected)


def test_web_text_and_shared_instructions_reach_one_model_call(playground):
    client, backend, _ = playground
    spec = request_spec()
    spec["instructions"] = "Only use explicit evidence.\nRetain this entire instruction."
    spec["questions"]["grade"] = {
        "type": "score",
        "instructions": "Grade the scene",
        "criteria": ["None", "Some", "Many"],
    }
    spec["questions"]["present"] = {
        "type": "independent",
        "instructions": "Check presence",
        "criteria": {"cat": "A cat", "dog": "A dog"},
    }
    response = client.post("/api/run", data={"spec": json.dumps(spec)})
    assert response.status_code == 200
    result = response.json()
    assert result["answers"]["animal"]["choice"] == "cat"
    assert result["answers"]["grade"]["type"] == "score"
    assert set(result["answers"]["present"]["probabilities"]) == {"cat", "dog"}
    assert len(backend.calls) == 1
    assert len(backend.calls[0][1]) == 4
    assert all(spec["instructions"] in request.instructions for request in backend.calls[0][1])


def test_uploaded_image_is_available_during_run_then_removed(playground):
    client, backend, directory = playground
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format="PNG")
    spec = request_spec()
    spec["media"] = [{"kind": "image", "name": "../../outside.png"}]
    response = client.post(
        "/api/run",
        data={"spec": json.dumps(spec)},
        files={"media": ("../../outside.png", image.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["media"][0]["width"] == 8
    assert "attachment-0.png" in backend.calls[0][0].images[0]
    assert list(directory.iterdir()) == []


def test_multi_picture_phone_jpeg_uses_full_resolution_primary_photo(playground):
    client, backend, directory = playground
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(
        image, format="MPO", save_all=True, append_images=[Image.new("RGB", (16, 12), "gray")]
    )
    spec = request_spec()
    spec["media"] = [{"kind": "image", "name": "phone.jpeg"}]
    response = client.post(
        "/api/run",
        data={"spec": json.dumps(spec)},
        files={"media": ("phone.jpeg", image.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    metadata = response.json()["media"][0]
    assert metadata["embedded_images"] == 2
    assert metadata["used_frame"] == 0
    assert (metadata["width"], metadata["height"]) == (32, 24)
    assert backend.calls[0][0].images[0].endswith("-primary.png")
    assert list(directory.iterdir()) == []


def test_choice_names_need_no_descriptions_and_grades_need_no_rubric(playground):
    client, backend, _ = playground
    spec = {
        "text": "A Porsche car",
        "questions": {
            "brand": {
                "type": "choice",
                "instructions": "Which brand is this?",
                "criteria": ["Porsche", "Mercedes"],
            },
            "quality": {"type": "score", "instructions": "How well maintained is it?", "levels": 5},
            "tags": {
                "type": "independent",
                "instructions": "Which are present?",
                "criteria": {"car": "", "person": " "},
            },
        },
    }
    response = client.post("/api/run", data={"spec": json.dumps(spec)})
    assert response.status_code == 200
    requests = backend.calls[0][1]
    assert requests[0].criteria == (("Porsche", "Porsche"), ("Mercedes", "Mercedes"))
    assert len(requests[1].criteria) == 5
    assert "lowest" in requests[1].criteria[0][1]
    assert response.json()["answers"]["quality"]["type"] == "score"


def test_malformed_media_is_rejected_and_cleaned(playground):
    client, backend, directory = playground
    spec = request_spec()
    spec["media"] = [{"kind": "image", "name": "bad.png"}]
    response = client.post(
        "/api/run",
        data={"spec": json.dumps(spec)},
        files={"media": ("bad.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert "Could not read this image" in response.json()["error"]
    assert backend.calls == []
    assert list(directory.iterdir()) == []


def test_browser_cannot_submit_local_paths(playground):
    client, backend, _ = playground
    spec = request_spec()
    spec["state"] = {"images": ["/private/file.png"]}
    response = client.post("/api/run", data={"spec": json.dumps(spec)})
    assert response.status_code == 400
    assert backend.calls == []


def test_cross_origin_requests_are_rejected(playground):
    client, backend, _ = playground
    response = client.post(
        "/api/run",
        data={"spec": json.dumps(request_spec())},
        headers={"Origin": "https://example.com"},
    )
    assert response.status_code == 403
    assert backend.calls == []


def test_schema_editor_rejects_duplicates_and_accepts_ordered_grades(playground):
    client, _, _ = playground
    response = client.post("/api/validate", content='{"questions":{"a":{},"a":{}}}')
    assert response.status_code == 400
    assert "Duplicate JSON key" in response.json()["error"]
    schema = {
        "questions": {
            "grade": {
                "type": "score",
                "instructions": "Grade this",
                "criteria": ["Low\nwith details", "High"],
            }
        }
    }
    response = client.post("/api/validate", json=schema)
    assert response.status_code == 200
    assert response.json() == schema


def test_empty_state_is_rejected_without_inference(playground):
    client, backend, _ = playground
    spec = request_spec()
    spec["text"] = ""
    response = client.post("/api/run", data={"spec": json.dumps(spec)})
    assert response.status_code == 400
    assert backend.calls == []


def test_independent_expansion_is_bounded():
    spec = request_spec()
    spec["questions"] = {
        "labels": {
            "type": "independent",
            "instructions": "Check each",
            "criteria": {str(i): "An option" for i in range(65)},
        }
    }
    with pytest.raises(ValueError, match="64 individual fields"):
        read_spec(json.dumps(spec))


def test_static_application_is_served(playground):
    client, _, _ = playground
    assert "Gemma E2B RLCD" in client.get("/").text
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "Record speech" in response.text


def wav_bytes(seconds):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(16000 * seconds))
    return data.getvalue()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for recording conversion"
)
def test_durationless_webm_recording_is_decoded_in_full(playground):
    client, backend, directory = playground
    recording = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "pipe:1",
        ],
        input=wav_bytes(0.2),
        capture_output=True,
        check=True,
    ).stdout
    spec = request_spec()
    spec["text"] = ""
    spec["media"] = [{"kind": "video", "name": "recording.webm"}]
    response = client.post(
        "/api/run",
        data={"spec": json.dumps(spec)},
        files={"media": ("recording.webm", recording, "video/webm")},
    )
    assert response.status_code == 200
    assert response.json()["media"][0]["kind"] == "audio"
    assert response.json()["media"][0]["duration_seconds"] == pytest.approx(0.2, abs=0.03)
    assert backend.calls[0][0].audio[0].endswith("-audio.wav")
    assert not backend.calls[0][0].videos
    assert list(directory.iterdir()) == []


@pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="ffprobe is required for audio validation"
)
def test_long_audio_is_rejected_without_truncation(playground):
    client, backend, directory = playground
    spec = request_spec()
    spec["media"] = [{"kind": "audio", "name": "long.wav"}]
    response = client.post(
        "/api/run",
        data={"spec": json.dumps(spec)},
        files={"media": ("long.wav", wav_bytes(31), "audio/wav")},
    )
    assert response.status_code == 400
    assert "30 seconds" in response.json()["error"]
    assert backend.calls == []
    assert list(directory.iterdir()) == []


def test_concurrent_requests_do_not_overlap_on_model(playground):
    client, backend, _ = playground
    started, release = Event(), Event()
    original = backend.score_batch

    def slow_score(state, requests):
        started.set()
        assert release.wait(5)
        return original(state, requests)

    backend.score_batch = slow_score
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/api/run", data={"spec": json.dumps(request_spec())})
        try:
            assert started.wait(5)
            assert client.get("/api/status").json()["busy"]
            second = client.post("/api/run", data={"spec": json.dumps(request_spec())})
            assert second.status_code == 429
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200
    assert len(backend.calls) == 1


def test_comparison_runs_each_path_once_on_same_uploaded_input(playground, monkeypatch):
    from pathlib import Path

    from gemma_rlcd import comparison

    client, backend, directory = playground
    normal_calls = []

    def generate(used_backend, state, questions):
        assert used_backend is backend
        assert Path(state.images[0]).is_file()
        normal_calls.append((state, questions))
        return {
            "answers": {"animal": "dog"},
            "raw_text": '{"animal":"dog"}',
            "valid": True,
            "error": None,
            "inference_seconds": 0.2,
            "output_tokens": 8,
        }

    monkeypatch.setattr(comparison, "generate_answers", generate)
    spec = request_spec()
    spec["media"] = [{"kind": "image", "name": "photo.png"}]
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format="PNG")
    response = client.post(
        "/api/compare",
        data={"spec": json.dumps(spec)},
        files={"media": ("photo.png", image.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    details = data["comparison"]
    assert len(backend.calls) == len(normal_calls) == 1
    assert backend.calls[0][0] == normal_calls[0][0]
    assert details["agreement"] == {"animal": False}
    assert details["parallel_values"] == {"animal": "cat"}
    assert details["seconds"]["normal"] == pytest.approx(0.2 + data["media_prepare_seconds"])
    assert details["seconds"]["parallel"] == pytest.approx(
        data["decision_seconds"] + data["media_prepare_seconds"]
    )
    assert details["normal_over_parallel"] > 0
    assert data["answers"]["animal"]["choice"] == "cat"
    assert list(directory.iterdir()) == []


def test_invalid_generated_answer_remains_visible_without_speedup_claim(playground, monkeypatch):
    from gemma_rlcd import comparison

    client, _, _ = playground
    monkeypatch.setattr(
        comparison,
        "generate_answers",
        lambda *args: {
            "answers": None,
            "raw_text": "not json",
            "valid": False,
            "error": "Invalid JSON",
            "inference_seconds": 0.1,
            "output_tokens": 2,
        },
    )
    response = client.post("/api/compare", data={"spec": json.dumps(request_spec())})
    assert response.status_code == 200
    data = response.json()
    assert data["comparison"]["normal_over_parallel"] is None
    assert data["comparison"]["agreement"]["animal"] is None
    assert data["comparison"]["normal"]["raw_text"] == "not json"
    assert data["answers"]["animal"]["choice"] == "cat"
