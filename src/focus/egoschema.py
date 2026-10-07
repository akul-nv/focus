"""Streaming EgoSchema-style five-choice QA from local videos and question JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .evaluation import (
    add_run_arguments,
    config_from_args,
    load_pipelines,
    run_video,
    runtime_metadata,
    write_json,
)


def extract_answer(text: str) -> int | None:
    """Read an explicit leading A-E answer, without guessing from arbitrary prose."""
    match = re.match(
        r"^\s*(?:(?:THE\s+)?ANSWER(?:\s+IS)?\s*[:=-]?\s*)?"
        r"[\(\[]?([A-E])[\)\]]?(?:[.):]|\s*$)",
        text.upper(),
    )
    return ord(match.group(1)) - ord("A") if match else None


def load_questions(path, video_dir):
    """Validate rows: video, question, five options, optional integer answer (0-4), id."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Questions must be a nonempty JSON list")
    seen = set()
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Question {index}: expected an object")
        for field in ("video", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Question {index}: {field} must be nonempty text")
        if (
            not isinstance(row.get("options"), list)
            or len(row["options"]) != 5
            or not all(isinstance(option, str) and option.strip() for option in row["options"])
        ):
            raise ValueError(f"Question {index}: options must contain five nonempty strings")
        answer = row.get("answer")
        if answer is not None and (type(answer) is not int or answer not in range(5)):
            raise ValueError(f"Question {index}: answer must be an integer from 0 through 4")
        identifier = str(row.get("id", index))
        if identifier in seen:
            raise ValueError(f"Duplicate question id: {identifier}")
        seen.add(identifier)
        video_path = Path(video_dir) / row["video"]
        if not video_path.is_file():
            raise ValueError(f"Video does not exist: {video_path}")
        normalized.append({**row, "id": identifier, "video_path": str(video_path)})
    return normalized


def answer_question(vlm, captions, representatives, question, options, *, max_frames=16):
    """Use the same narrative and representative-frame QA stage for every method."""
    if max_frames < 1 or not representatives:
        raise ValueError("QA requires representative frames and a positive max_frames")
    if len(representatives) > max_frames:
        indices = np.linspace(0, len(representatives) - 1, max_frames, dtype=int)
        representatives = [representatives[i] for i in indices]
    narrative = "\n".join(f"[Observation {i + 1}] {text}" for i, text in enumerate(captions))
    choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(options))
    prompt = (
        f"You have been watching a video. Here is a chronological summary:\n{narrative}\n\n"
        f"Based on everything observed, answer the following question.\n"
        f"Question: {question}\n{choices}\n\n"
        "Answer with ONLY the letter (A, B, C, D, or E). Do not explain."
    )
    response, ms = vlm.generate_from_frames(
        representatives,
        prompt=prompt,
        max_new_tokens=30,
        do_sample=False,
    )
    return response, ms, len(representatives)


def summarize(results):
    labeled = [row for row in results if row.get("answer") is not None]
    correct = sum(row.get("prediction") == row["answer"] for row in labeled)
    return {
        "questions_completed": len(results),
        "labeled_questions": len(labeled),
        "correct": correct,
        "accuracy": correct / len(labeled) if labeled else None,
        "unparsed_predictions": sum(row.get("prediction") is None for row in results),
        "caption_calls": sum(row["stream"]["stats"]["caption_calls"] for row in results),
        "qa_calls": len(results),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--max-qa-frames", type=int, default=16)
    add_run_arguments(parser, target_fps=6.0)
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if args.max_qa_frames < 1:
            raise ValueError("max-qa-frames must be positive")
        if args.output.exists() and not args.overwrite:
            raise ValueError("Output already exists; choose a new path or pass --overwrite")
        questions = load_questions(args.questions, args.video_dir)
        inputs = [args.questions, *(Path(question["video_path"]) for question in questions)]
        if any(path.resolve() == args.output.resolve() for path in inputs):
            raise ValueError("Output must not overwrite an input file")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    pipeline, encoder, base = load_pipelines(args)
    output = {
        "schema_version": 1,
        "arguments": vars(args),
        "results": [],
        "questions_requested": len(questions),
        "complete": False,
        "runtime": runtime_metadata(base),
    }
    for question in questions:
        stream, representatives = run_video(
            question["video_path"],
            pipeline,
            config,
            encoder=encoder,
            collect_representatives=True,
        )
        response, qa_ms, qa_frames = answer_question(
            base,
            [row["text"] for row in stream["responses"]],
            representatives,
            question["question"],
            question["options"],
            max_frames=args.max_qa_frames,
        )
        output["results"].append(
            {
                **question,
                "stream": stream,
                "response": response,
                "prediction": extract_answer(response),
                "qa_time_ms": qa_ms,
                "qa_frames": qa_frames,
            }
        )
        output["summary"] = summarize(output["results"])
        output["complete"] = len(output["results"]) == len(questions)
        write_json(args.output, output)
        print(f"{question['id']}: {output['summary']} -> {args.output}")


if __name__ == "__main__":
    main()
