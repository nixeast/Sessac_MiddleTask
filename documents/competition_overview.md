# Kaggle Playground Series S6E10 — Predicting Airline Satisfaction

대회 링크: https://www.kaggle.com/competitions/playground-series-s6e10

## 한 줄 요약
항공기 승객의 정보와 서비스 평가 점수를 보고 **승객이 만족했는지(`satisfaction`) 확률을 예측**하는 **이진 분류** 대회.

## 대회 성격
- Kaggle **Playground Series**: 매달 열리는 입문·연습용 대회 (시즌 6, 10번째 에피소드)
- 데이터는 공개 데이터셋 *Airline Passenger Satisfaction*(약 13만 행)을 바탕으로 **딥러닝 모델이 합성(synthetic)한 데이터**
  - 원본보다 훨씬 크고, 원본과 분포가 조금 다를 수 있음
  - 원본 데이터를 학습에 추가로 쓰는 것도 흔히 쓰는 전략

## 데이터 (`data/`)
| 파일 | 행 수 | 설명 |
|---|---|---|
| `train.csv` | 699,635 | 피처 + 타깃 `satisfaction` |
| `test.csv` | 299,844 | 피처만 있음 (id 699635~) |
| `sample_submission.csv` | 299,844 | 제출 양식 |

### 피처 (22개, `id` 제외)
- **수치형**: `Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`
- **서비스 평가 점수 (0~5점, 14개)**: `Inflight wifi service`, `Departure/Arrival time convenient`, `Ease of Online booking`, `Gate location`, `Food and drink`, `Online boarding`, `Seat comfort`, `Inflight entertainment`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Cleanliness`
- **범주형**: `Gender`, `Customer Type`(Loyal / disloyal), `Type of Travel`(Business / Personal), `Class`(Business / Eco / Eco Plus)

### 타깃
- `satisfaction`: `True`(만족) / `False`(중립 또는 불만족)
- 비율: False 약 55.6% (389,296) / True 약 44.4% (310,339) → 크게 불균형하지 않음

### 참고할 점
- `Arrival Delay in Minutes`에 결측치가 있음 (train 292건)
- 지연 시간 컬럼은 오른쪽으로 긴 꼬리 분포 → 로그 변환 등을 고려
- 평가 점수의 0은 "해당 없음/미응답"일 가능성이 있음

## 해야 할 일
1. `train.csv`로 모델 학습 (타깃 `satisfaction`을 0/1로 변환)
2. `test.csv`의 각 `id`에 대해 **만족할 확률(0~1)** 예측
3. 아래 형식으로 CSV를 만들어 제출

```csv
id,satisfaction
699635,0.83
699636,0.12
...
```

> 클래스(True/False)가 아니라 **확률값**을 제출해야 함.

## 평가 지표
- 제출 형식이 확률값이고 제공된 베이스라인 노트북도 AUC로 검증하고 있어, **ROC AUC**로 평가하는 것으로 보임 (높을수록 좋음)
- 정확한 지표와 마감일은 대회 페이지의 *Overview → Evaluation / Timeline*에서 한 번 더 확인할 것

## 추천 진행 순서
1. **EDA**: 타깃 분포, 결측치, 피처별 만족률 확인
2. **베이스라인**: `baseline[Lab].ipynb`, `mlp_pytorch_baseline[Lab].ipynb` 참고
3. **검증 전략**: Stratified K-Fold(5-fold)로 OOF AUC 측정 → 리더보드보다 CV를 믿기
4. **모델**: LightGBM / XGBoost / CatBoost 같은 GBDT가 정형 데이터에서 강함, MLP는 앙상블용
5. **개선**: 피처 엔지니어링(서비스 점수 합·평균, 지연 시간 차이 등), 원본 데이터 추가, 하이퍼파라미터 튜닝, 여러 모델 앙상블
