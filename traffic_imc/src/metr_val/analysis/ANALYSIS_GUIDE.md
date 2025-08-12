# RNN 모델 성능 분석 가이드

이 가이드는 훈련된 RNN 모델의 성능을 다각도로 분석하는 방법을 설명합니다.

## 빠른 시작

### 1. 의존성 설치
```bash
# 새로운 의존성들을 설치
pdm install
```

### 2. 전체 분석 실행
```bash
# 기본 사용법 (가장 간단)
python analyze_model_performance.py --model_path ./output/rnn/best-epoch=14-val_loss=0.00.ckpt

# 상세 옵션 지정
python analyze_model_performance.py \
    --model_path ./output/rnn/best-epoch=14-val_loss=0.00.ckpt \
    --data_path ./data/selected_small_v1/metr-imc.h5 \
    --results_dir ./my_analysis_results \
    --training_logs ./wandb \
    --n_samples 1500
```

## 분석 결과물

### 📊 정적 시각화 (PNG 파일)
분석 실행 후 `analysis_results/plots/` 폴더에 생성됩니다:

1. **time_series_comparison.png**: 실제값 vs 예측값 시계열 비교
2. **error_analysis.png**: 오차 분포, 잔차 분석, Q-Q 플롯
3. **temporal_analysis.png**: 시간대별/요일별 예측 성능 패턴
4. **training_curves.png**: 훈련/검증 손실 곡선 (로그가 있는 경우)
5. **convergence_analysis.png**: 수렴 패턴 및 학습률 분석

### 🌐 인터랙티브 대시보드 (HTML 파일)
`analysis_results/interactive/` 폴더에 생성됩니다:

1. **dashboard.html**: 예측 성능 인터랙티브 대시보드
2. **training_dashboard.html**: 훈련 과정 인터랙티브 분석

### 💾 데이터 파일
`analysis_results/data/` 폴더에 저장됩니다:

1. **predictions.pkl**: 모든 예측 결과와 메타데이터

## 세부 분석 방법

### 개별 분석 스크립트 사용

```python
# 1. 예측 성능 분석만 실행
from src.metr_val.visualization_analysis import analyze_rnn_predictions

analyzer = analyze_rnn_predictions(
    model_checkpoint_path="./output/rnn/best-epoch=14-val_loss=0.00.ckpt",
    data_path="./data/selected_small_v1/metr-imc.h5",
    results_dir="./analysis_results"
)

# 2. 훈련 과정 분석만 실행
from src.metr_val.training_diagnostics import analyze_training_progress

convergence_info = analyze_training_progress(
    log_path="./wandb",  # 또는 특정 CSV 파일 경로
    results_dir="./analysis_results"
)
```

### 사용자 정의 분석

```python
# 예측 데이터 로드하여 커스텀 분석
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# 저장된 예측 결과 로드
with open('./analysis_results/data/predictions.pkl', 'rb') as f:
    predictions = pickle.load(f)

# 커스텀 분석 예시
actuals = predictions['actuals_original']
preds = predictions['predictions_original']
timestamps = predictions['timestamps']

# 특정 기간 분석
specific_period = (timestamps >= '2024-01-01') & (timestamps <= '2024-01-07')
period_actuals = actuals[specific_period]
period_preds = preds[specific_period]

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(timestamps[specific_period], period_actuals, label='Actual', alpha=0.8)
plt.plot(timestamps[specific_period], period_preds, label='Predicted', alpha=0.8)
plt.legend()
plt.title('Custom Period Analysis')
plt.show()
```

## 분석 결과 해석 가이드

### 1. 시계열 비교 (time_series_comparison.png)
- **상단 플롯**: 스케일된 데이터에서의 예측 성능
- **중간 플롯**: 원본 스케일에서의 예측 성능
- **하단 플롯**: 시간에 따른 잔차 패턴

**해석 포인트**:
- 예측선과 실제선이 얼마나 일치하는가?
- 특정 시간대에 예측이 어려운 패턴이 있는가?
- 잔차에 체계적인 패턴이 있는가? (이상적으로는 무작위여야 함)

### 2. 오차 분석 (error_analysis.png)
- **잔차 히스토그램**: 정규분포에 가까워야 함
- **Q-Q 플롯**: 직선에 가까울수록 정규분포
- **잔차 vs 예측값**: 패턴이 없어야 함 (heteroscedasticity 확인)
- **교통량 구간별 오차**: 어떤 교통량 수준에서 예측이 어려운가?

### 3. 시간 패턴 분석 (temporal_analysis.png)
- **시간대별 오차**: 출퇴근 시간의 예측 난이도
- **요일별 오차**: 주중/주말 예측 성능 차이
- **히트맵**: 특정 시간대-요일 조합에서의 어려움

### 4. 훈련 과정 분석
- **수렴 패턴**: 모델이 제대로 학습했는가?
- **과적합 징후**: 검증 손실이 증가하는 구간이 있는가?
- **학습률 영향**: 학습률 조정이 적절했는가?

## 문제 해결

### 일반적인 문제들

1. **예측이 실제값을 따라가지 못함**
   - 모델 복잡도 증가 고려
   - 더 많은 훈련 데이터 필요
   - Feature engineering 개선

2. **특정 시간대에서 성능 저하**
   - 해당 시간대 데이터의 패턴 분석
   - 추가 시간 특성 (휴일, 특수 이벤트) 고려

3. **과적합 징후**
   - 정규화 기법 추가
   - 드롭아웃 비율 증가
   - 조기 중단 기준 조정

4. **수렴하지 않는 훈련**
   - 학습률 조정
   - 배치 크기 변경
   - 모델 아키텍처 단순화

## 다음 단계

분석 결과를 바탕으로:

1. **모델 개선**: 발견된 문제점을 해결하는 새로운 모델 훈련
2. **STGCN 비교**: 기존 RNN과 STGCN 모델 성능 비교
3. **앙상블**: 여러 모델의 예측을 결합하여 성능 향상
4. **배포 준비**: 실시간 예측 시스템 구축

## 추가 도움

- 코드 문제: GitHub Issues 또는 코드 주석 참조
- 결과 해석: 도메인 전문가와 상의
- 성능 개선: 머신러닝 커뮤니티 베스트 프랙티스 참조