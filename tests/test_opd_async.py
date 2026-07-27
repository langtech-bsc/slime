import math
import time
from types import SimpleNamespace

import pytest
import torch

from slime.utils.opd import (
    OpdHiddenStateBuffer,
    OpdHiddenStatePayload,
    hidden_topk_kl,
    teacher_topk_from_hidden,
    topk_kl,
)
from slime.rollout.opd_prompt import build_teacher_prompt, salamandra_privileged_teacher_prompt
from slime.utils.types import Sample


def test_teacher_topk_streaming_matches_dense():
    torch.manual_seed(1)
    hidden = torch.randn(5, 7)
    weight = torch.randn(19, 7)
    values, indices = teacher_topk_from_hidden(
        hidden,
        weight,
        top_k=4,
        temperature=1.2,
        vocab_chunk_size=3,
    )
    reference = torch.log_softmax(hidden @ weight.t() / 1.2, dim=-1)
    expected_values, expected_indices = torch.topk(reference, k=4, dim=-1)
    assert torch.equal(indices, expected_indices)
    assert torch.allclose(values, expected_values, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_hidden_topk_kl_matches_dense_reference(direction):
    torch.manual_seed(2)
    student_hidden = torch.randn(4, 6, requires_grad=True)
    teacher_hidden = torch.randn(4, 6)
    student_weight = torch.randn(13, 6, requires_grad=True)
    teacher_weight = torch.randn(13, 6)
    valid = torch.tensor([True, True, False, True])

    loss, _ = hidden_topk_kl(
        student_hidden,
        student_weight,
        teacher_hidden,
        teacher_weight,
        valid,
        top_k=5,
        temperature=0.9,
        direction=direction,
        normalize_teacher=True,
        pointwise_clip=0,
        token_chunk_size=2,
        vocab_chunk_size=4,
    )

    teacher_logps = torch.log_softmax(teacher_hidden @ teacher_weight.t() / 0.9, dim=-1)
    teacher_values, teacher_ids = torch.topk(teacher_logps, k=5, dim=-1)
    student_logps = torch.log_softmax(student_hidden @ student_weight.t() / 0.9, dim=-1)
    selected_student = torch.gather(student_logps, dim=-1, index=teacher_ids)
    reference, _ = topk_kl(
        selected_student,
        teacher_values,
        torch.ones_like(teacher_ids, dtype=torch.bool),
        direction=direction,
        normalize_teacher=True,
        pointwise_clip=0,
    )
    reference = reference.masked_fill(~valid, 0)
    assert torch.allclose(loss, reference, atol=1e-5, rtol=1e-5)

    loss.sum().backward()
    assert torch.isfinite(student_hidden.grad).all()
    assert torch.isfinite(student_weight.grad).all()


def test_reverse_kl_normalizes_student_on_teacher_support():
    student = torch.tensor([[-5.0, -2.0, -3.0]])
    teacher = torch.tensor([[-1.0, -2.0, -4.0]])
    loss, _ = topk_kl(
        student,
        teacher,
        torch.ones_like(student, dtype=torch.bool),
        direction="reverse",
        normalize_teacher=True,
        pointwise_clip=0,
    )
    student_norm = torch.log_softmax(student, dim=-1)
    teacher_norm = torch.log_softmax(teacher, dim=-1)
    expected = (student_norm.exp() * (student_norm - teacher_norm)).sum(dim=-1)
    assert torch.allclose(loss, expected)


def test_reverse_kl_always_normalizes_teacher_support():
    student = torch.tensor([[-5.0, -2.0, -3.0]])
    teacher = torch.tensor([[-1.0, -2.0, -4.0]])
    valid = torch.ones_like(student, dtype=torch.bool)
    normalized, _ = topk_kl(
        student,
        teacher,
        valid,
        direction="reverse",
        normalize_teacher=True,
        pointwise_clip=0,
    )
    requested_unnormalized, _ = topk_kl(
        student,
        teacher,
        valid,
        direction="reverse",
        normalize_teacher=False,
        pointwise_clip=0,
    )
    assert torch.allclose(requested_unnormalized, normalized)


def test_hidden_buffer_is_bounded_and_delete_on_pop():
    buffer = OpdHiddenStateBuffer(max_pending_batches=1)
    payload = OpdHiddenStatePayload(
        request_id="a",
        hidden_states=torch.zeros(2, 3),
        valid_mask=torch.ones(2, dtype=torch.bool),
        created_at=time.time() - 1,
    )
    buffer.put(payload)
    with pytest.raises(ValueError, match="Duplicate"):
        buffer.put(payload)
    with pytest.raises(BufferError, match="full"):
        buffer.put(
            OpdHiddenStatePayload(
                request_id="b",
                hidden_states=torch.zeros(1),
                valid_mask=torch.ones(1, dtype=torch.bool),
                created_at=time.time(),
            )
        )
    metrics = buffer.metrics()
    assert metrics["opd/cache_pending"] == 1
    assert metrics["opd/cache_bytes"] == payload.nbytes
    assert metrics["opd/cache_oldest_age_seconds"] >= 1
    assert buffer.pop("a") is payload
    with pytest.raises(KeyError, match="Unknown"):
        buffer.pop("a")


def test_teacher_prompt_modes():
    sample = Sample(prompt="What is 2+2?", label="4", response="4")
    trajectory_args = SimpleNamespace(
        opd_teacher_prompt_mode="trajectory",
        opd_teacher_prompt_function_path=None,
    )
    assert build_teacher_prompt(trajectory_args, sample) == sample.prompt

    custom_args = SimpleNamespace(
        opd_teacher_prompt_mode="custom",
        opd_teacher_prompt_function_path=(
            "slime.rollout.opd_prompt.salamandra_privileged_teacher_prompt"
        ),
    )
    prompt = build_teacher_prompt(custom_args, sample)
    assert prompt == salamandra_privileged_teacher_prompt(sample)
    assert "Verified reference solution:\n4" in prompt[0]["content"]


def test_topk_mass_is_unrenormalized_teacher_mass():
    teacher = torch.log(torch.tensor([[0.5, 0.25]]))
    _, stats = topk_kl(
        torch.log(torch.tensor([[0.4, 0.3]])),
        teacher,
        torch.ones_like(teacher, dtype=torch.bool),
        direction="forward",
        normalize_teacher=True,
        pointwise_clip=0,
    )
    assert math.isclose(float(stats["topk_mass"]), 0.75, abs_tol=1e-6)
