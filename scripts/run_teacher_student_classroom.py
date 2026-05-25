#!/usr/bin/env python
"""Run teacher-student harness evolution on a small HumanEval slice."""

from __future__ import annotations

import argparse

from classroom.benchmark import HumanEvalBenchmark
from classroom.factory import make_chat_model, make_students, make_teacher
from classroom.interaction import TeacherStudentClassroom
from classroom.model import load_model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.json", help="JSON file with base_url/api_key/model names")
    parser.add_argument("--model", default=None, help="OpenAI-compatible model for students")
    parser.add_argument("--teacher-model", default=None, help="Optional OpenAI-compatible teacher model")
    parser.add_argument("--rule-teacher", action="store_true", help="Use deterministic rule-based teacher advice")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-s", type=int, default=None, help="HTTP timeout for each model call")
    parser.add_argument("--students", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--output", default="runs/classroom/rounds.jsonl")
    parser.add_argument("--log", default="runs/classroom/transcript.md")
    args = parser.parse_args()

    config = None
    if args.config:
        try:
            config = load_model_config(args.config)
        except FileNotFoundError:
            if args.model is None:
                raise

    model_name = args.model or (config.student_model if config else None)
    if model_name is None:
        raise ValueError("Provide --model or create configs/model.local.json.")
    teacher_name = None if args.rule_teacher else args.teacher_model or (config.teacher_model if config else None)
    base_url = args.base_url or (config.base_url if config else None)
    api_key = args.api_key or (config.api_key if config else None)
    temperature = config.temperature if config else 0.0
    timeout_s = args.timeout_s if args.timeout_s is not None else (config.timeout_s if config else 60)

    student_model = make_chat_model(
        model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        timeout_s=timeout_s,
    )
    teacher_model = None
    if teacher_name:
        teacher_model = make_chat_model(
            teacher_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            timeout_s=timeout_s,
        )

    benchmark = HumanEvalBenchmark(task_ids=[f"HumanEval/{i}" for i in range(args.tasks)])
    classroom = TeacherStudentClassroom(
        benchmark=benchmark,
        teacher=make_teacher(teacher_model),
        students=make_students(student_model, count=args.students),
        output_path=args.output,
        log_path=args.log,
    )
    report = classroom.run(generations=args.generations, tasks_per_generation=args.tasks)
    print(
        f"rounds={len(report.rounds)} "
        f"before={report.average_before:.3f} "
        f"after={report.average_after:.3f} "
        f"gain={report.gain:+.3f} "
        f"output={args.output} "
        f"log={args.log}"
    )


if __name__ == "__main__":
    main()
