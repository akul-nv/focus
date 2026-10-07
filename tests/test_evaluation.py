"""Scheduling, feedback and QA scoring without external datasets or model weights."""

import json

import numpy as np
import pytest
import torch

from focus import egoschema, evaluation
from focus.evaluation import RunConfig, run_stream


class FakeVLM:
    extract_attention = False
    last_attention_map = None
    last_grid_hw = (1, 2)

    def __init__(self):
        self.calls = []

    def generate_from_frames(self, frames, prompt, **kwargs):
        self.calls.append((frames, prompt, kwargs))
        self.last_attention_map = torch.tensor([[0.2, 1.0]])
        return f"observation {len(self.calls)}", 2.0

    def encode_frame(self, frame):
        angle = int(frame[0, 0, 0]) * 0.3
        return torch.tensor([[np.cos(angle), np.sin(angle)], [1.0, 0.0]], dtype=torch.float32)


def stream(count):
    return [(i, i / 24, np.full((2, 2, 3), i, dtype=np.uint8)) for i in range(count)]


def test_uniform_cadence_and_partial_burst_flush():
    vlm = FakeVLM()
    result, reps = run_stream(
        stream(11),
        vlm,
        RunConfig(method="uniform", burst_size=3, interval=5),
        collect_representatives=True,
    )
    assert [row["frame_indices"] for row in result["responses"]] == [[0, 1, 2], [5, 6, 7], [10]]
    assert result["stats"]["caption_calls"] == 3
    assert result["stats"]["frames_sent_to_vlm"] == 7
    assert len(reps) == 3
    assert "observation 1" in vlm.calls[1][1]


def test_focus_feedback_is_available_for_next_burst_and_reset_between_videos():
    vlm = FakeVLM()
    config = RunConfig(method="focus", burst_size=2, threshold=0, prompt="Identify the robot.")
    result, _ = run_stream(stream(5), vlm, config, encoder=vlm)
    assert len(result["responses"]) == 3
    assert vlm.calls[0][2]["spatial_prior"] is None
    assert vlm.calls[1][2]["spatial_prior"] is not None
    assert "Identify the robot." in vlm.calls[1][1]
    assert vlm.extract_attention is False
    run_stream(stream(2), vlm, config, encoder=vlm)
    assert vlm.calls[-1][2]["spatial_prior"] is None


def test_gating_only_omits_spatial_prior_and_zero_frame_input_is_an_error():
    vlm = FakeVLM()
    run_stream(stream(2), vlm, RunConfig(method="gating", burst_size=2), encoder=vlm)
    assert "spatial_prior" not in vlm.calls[0][2]
    with pytest.raises(ValueError, match="no decodable"):
        run_stream([], vlm, RunConfig(method="uniform"))
    with pytest.raises(ValueError, match="interval"):
        RunConfig(method="uniform", burst_size=32, interval=20)


def test_video_reader_uses_fractional_schedule_and_releases(monkeypatch):
    class Capture:
        released = False
        i = -1

        def isOpened(self):
            return True

        def get(self, prop):
            return 25.0

        def read(self):
            self.i += 1
            return self.i < 25, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(evaluation.cv2, "VideoCapture", lambda path: capture)
    frames = list(evaluation.iter_video_frames("fake", 6))
    assert [row[0] for row in frames] == [0, 5, 9, 13, 17, 21]
    assert frames[-1][1] == 21 / 25
    assert capture.released


@pytest.mark.parametrize("module", [evaluation, egoschema])
def test_help_does_not_load_models(module, monkeypatch, capsys):
    monkeypatch.setattr(module, "load_pipelines", lambda args: pytest.fail("loaded model"))
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "--method" in capsys.readouterr().out


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", 0),
        ("Answer: C", 2),
        ("(D)", 3),
        ("B. explanation", 1),
        ("A or B", None),
        ("A person is cooking", None),
        ("", None),
    ],
)
def test_answer_parser_does_not_guess_from_articles(text, expected):
    assert egoschema.extract_answer(text) == expected


def test_scoring_counts_invalid_answers_and_excludes_unlabeled():
    rows = [
        {"answer": 1, "prediction": pred, "stream": {"stats": {"caption_calls": 2}}}
        for pred in (1, None, 0)
    ]
    rows.append({"prediction": 2, "stream": {"stats": {"caption_calls": 2}}})
    summary = egoschema.summarize(rows)
    assert summary["labeled_questions"] == 3
    assert summary["accuracy"] == 1 / 3
    assert summary["unparsed_predictions"] == 1
    assert summary["qa_calls"] == 4


def test_questions_validate_options_and_missing_video_before_loading_models(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    questions = tmp_path / "questions.json"
    row = {"video": video.name, "question": "What?", "options": list("ABCDE"), "answer": 4}
    questions.write_text(json.dumps([row]))
    assert egoschema.load_questions(questions, tmp_path)[0]["answer"] == 4
    row["answer"] = 5
    questions.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="integer"):
        egoschema.load_questions(questions, tmp_path)


def test_callable_encoder_and_gating_disable_existing_attention():
    vlm = FakeVLM()
    vlm.extract_attention = True
    observations = []

    def encode(frame):
        observations.append(vlm.extract_attention)
        return torch.ones(2, 2)

    run_stream(stream(2), vlm, RunConfig(method="gating", burst_size=2), encoder=encode)
    assert observations == [False, False]
    assert vlm.extract_attention is True


def test_cli_rejects_overwriting_source_before_model_load(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source data")
    monkeypatch.setattr(evaluation, "load_pipelines", lambda args: pytest.fail("loaded model"))
    with pytest.raises(SystemExit) as exc:
        evaluation.main([str(source), "--output", str(source), "--overwrite"])
    assert exc.value.code == 2
    assert source.read_bytes() == b"source data"
