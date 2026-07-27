from __future__ import annotations

from typing import Any


def salamandra_privileged_teacher_prompt(sample: Any) -> list[dict[str, str]]:
    if not sample.label:
        raise ValueError("Privileged OPD prompting requires Sample.label")
    if isinstance(sample.prompt, str):
        problem = sample.prompt
    else:
        user_messages = [message.get("content", "") for message in sample.prompt if message.get("role") == "user"]
        problem = "\n".join(str(content) for content in user_messages)
    return [
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Verified reference solution:\n{sample.label}\n\n"
                "Solve the problem using your own reasoning. "
                "Do not mention that you saw a reference solution."
            ),
        }
    ]


def build_teacher_prompt(args, sample: Any) -> str | list[dict[str, str]]:
    if args.opd_teacher_prompt_mode == "trajectory":
        return sample.prompt
    if args.opd_teacher_prompt_mode != "custom":
        raise ValueError(f"Unsupported OPD teacher prompt mode: {args.opd_teacher_prompt_mode}")
    from slime.utils.misc import load_function

    prompt_builder = load_function(args.opd_teacher_prompt_function_path)
    prompt = prompt_builder(sample)
    if not isinstance(prompt, (str, list)):
        raise TypeError("OPD teacher prompt function must return a string or chat-message list")
    return prompt
