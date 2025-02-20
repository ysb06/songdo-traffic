import random
from datetime import datetime, timedelta

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from songdo_llm.model.base.llm import LLMWeight


def generate_random_timestamps(start: datetime, end: datetime, num_samples: int = 8):
    """
    start부터 end 사이의 한 시간 간격 중에서
    랜덤하게 num_samples개의 타임스탬프를 뽑아 반환합니다.
    """
    total_hours = int((end - start).total_seconds() // 3600)
    chosen_hours = random.sample(range(total_hours + 1), k=num_samples)

    timestamps = []
    for h in chosen_hours:
        ts = start + timedelta(hours=h)
        timestamps.append(ts)
    return timestamps


@pytest.mark.parametrize("num_samples", [8])
def test_llm_model_smoke(num_samples):
    """
    모든 입력에 대한 목표값을 1로 설정하고 짧게 학습한 뒤,
    모델의 출력이 1에 근접하는지 확인하는 Smoke Test.
    """
    start_dt = datetime(2024, 3, 1, 0, 0, 0)
    end_dt = datetime(2024, 8, 31, 23, 0, 0)
    training_timestamps = generate_random_timestamps(
        start_dt, end_dt, num_samples=num_samples
    )
    test_timestamps = generate_random_timestamps(
        start_dt, end_dt, num_samples=num_samples
    )

    model = LLMWeight(
        llm_name="deepseek-r1:8b",
        classifier_name="microsoft/deberta-v3-base",
        llm_cache_path="./output/llm_cache",
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 3
    true_value = 5.0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        with tqdm(total=len(training_timestamps)) as pbar:
            for ts in training_timestamps:
                pbar.set_description(f"{ts}")
                logits = model(ts)

                target = torch.tensor([[true_value]], dtype=torch.float, requires_grad=False)
                loss: torch.Tensor = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                pbar.update(1)

        avg_loss = epoch_loss / len(training_timestamps)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    final_preds = []
    with torch.no_grad():
        for ts in tqdm(test_timestamps):
            y_pred: torch.Tensor = model(ts)
            final_preds.append(y_pred.item())

    mean_pred = sum(final_preds) / len(final_preds)
    print(f"Mean prediction after training: {mean_pred:.4f}")

    assert mean_pred > true_value - 1 and mean_pred < true_value + 1, "Mean prediction is not close enough to 1.0"
