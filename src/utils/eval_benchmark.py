"""Forward-pass latency micro-benchmarking for W&B."""

from __future__ import annotations

import time

import numpy as np
import torch


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure_mean_latency_us(
    model,
    dataloader,
    device: torch.device,
    warmup_iters: int = 10,
    measure_iters: int = 100,
) -> float:
    """Mean single-sample forward-pass latency (microseconds), batch size 1."""
    model.eval()
    x, _, _ = next(iter(dataloader))
    x = x[:1].to(device)

    with torch.inference_mode():
        for _ in range(warmup_iters):
            model(x)
        _sync_device(device)

        times_s: list[float] = []
        for _ in range(measure_iters):
            _sync_device(device)
            start = time.perf_counter()
            model(x)
            _sync_device(device)
            times_s.append(time.perf_counter() - start)

    return float(np.mean(times_s) * 1e6)


def collect_student_latency_metrics(
    student,
    teacher,
    test_loader,
    device: torch.device,
) -> dict[str, float]:
    """Return mean single-sample latency for student and teacher."""
    return {
        "latency/student_mean_us": measure_mean_latency_us(student, test_loader, device),
        "latency/teacher_mean_us": measure_mean_latency_us(teacher, test_loader, device),
    }


def collect_teacher_latency_metrics(
    model,
    test_loader,
    device: torch.device,
) -> dict[str, float]:
    """Return mean single-sample latency for teacher."""
    return {
        "latency/teacher_mean_us": measure_mean_latency_us(model, test_loader, device),
    }
