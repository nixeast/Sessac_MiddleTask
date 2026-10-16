# Kaggle Playground Series S6E10 — Airline Passenger Satisfaction 대회 요약

본 문서는 캐글(Kaggle) Playground Series Season 6 Episode 10 대회의 문제 정의, 데이터 구성, 베이스라인 분석 결과 및 개선 전략을 정리한 문서입니다.

- **대회 링크**: [Kaggle Playground Series S6E10 — Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)

---

## 1. 대회 개요 및 문제 정의

| 항목 | 내용 |
| :--- | :--- |
| **대회명** | Playground Series Season 6, Episode 10 |
| **작업 유형 (Task)** | 정형 데이터 이진 분류 (Binary Tabular Classification) |
| **목적 (Objective)** | 비행 운항 정보, 승객 인적사항, 기내 서비스 품질 평가 지표를 활용하여 승객의 최종 서비스 만족 여부(`satisfaction`) 예측 |
| **타깃 변수 (Target)** | `satisfaction`<br>- `satisfied` (1): 만족<br>- `neutral or dissatisfied` (0): 보통 또는 불만족 |
| **평가 지표 (Metric)** | **ROC-AUC (Area Under the ROC Curve)**<br>- 임계값(Threshold)에 독립적인 랭킹 기반 판별력 측정<br>- 클래스 간 상대적 순위(Ranking) 예측 확률 산출 |
| **제출 규격 (Submission)** | `id`(승객 식별자), `satisfaction`($0.0 \sim 1.0$ 사이의 만족 예측 확률) |

---

## 2. 데이터셋 구성 및 피처 스키마

제공 데이터는 `train.csv`, `test.csv`, `sample_submission.csv` 3가지 파일로 구성되어 있습니다.

### 피처 상세 분류 (총 22개 피처)

1. **식별자 & 타깃**
   - `id`: 각 승객의 고유 ID (학습에서 제외)
   - `satisfaction`: 타깃 레이블 (`train.csv`에만 존재)

2. **범주형 피처 (4개)**
   - `Gender`: 성별 (`Male`, `Female`)
   - `Customer Type`: 고객 유형 (`Loyal Customer`, `disloyal Customer`)
   - `Type of Travel`: 여행 목적 (`Business travel`, `Personal Travel`)
   - `Class`: 좌석 등급 (`Business`, `Eco`, `Eco Plus`)

3. **수치형 연속 변수 (4개)**
   - `Age`: 승객 연령
   - `Flight Distance`: 비행 거리
   - `Departure Delay in Minutes`: 출발 지연 시간 (분 단위)
   - `Arrival Delay in Minutes`: 도착 지연 시간 (분 단위)

4. **기내 서비스 세부 만족도 평가 피처 (14개, 0~5점 척도)**
   - `Inflight wifi service`
   - `Departure/Arrival time convenient`
   - `Ease of Online booking`
   - `Gate location`
   - `Food and drink`
   - `Online boarding`
   - `Seat comfort`
   - `Inflight entertainment`
   - `On-board service`
   - `Leg room service`
   - `Baggage handling`
   - `Checkin service`
   - `Inflight service`
   - `Cleanliness`
   *(참고: 0점은 해당 없음/미이용(Not Applicable) 또는 미응답 특성)*

---

## 3. 핵심 EDA (탐색적 데이터 분석) 인사이트

1. **타깃 클래스 분포**
   - Satisfied 약 44%, Neutral/Dissatisfied 약 56%로 극심한 불균형은 없어 오버/언더샘플링보다는 **Stratified Split(계층화 분할)**이 적합합니다.

2. **결측치(Missing Values)**
   - `Arrival Delay in Minutes` 피처에서만 소량의 결측치(전체 0.05% 미만) 발생.
   - 트리 기반 모델(LightGBM, XGBoost, CatBoost)은 기본 결측 처리 분기를 학습할 수 있고, 선형/신경망 모델은 중앙값(Median) 대체를 적용합니다.

3. **강력한 영향력을 가진 범주형 변수**
   - `Type of Travel` (출장 vs 개인 여행) 및 `Class` (Business vs Eco)가 만족도 결정에 가장 큰 영향력을 발휘합니다.
   - 반면 `Gender`는 성별 간 만족 비율 편차가 미미하여 피처 기여도가 매우 낮습니다.

4. **평점의 비선형성 및 0점 특이 구간**
   - $1 \sim 5$점 구간은 평점 상승에 따라 만족 비율이 급격히 증가합니다.
   - **0점 구간의 반전**: `Inflight wifi service`나 `Online boarding` 등 특정 피처는 0점일 때 만족 비율이 오히려 높게 나타나는 U-shape 형태의 비선형성을 보입니다(서비스 미이용/해당 없음 도메인 특성).
   - 따라서 선형 모델보다는 비선형 패턴을 학습할 수 있는 GBDT 모델이 유리합니다.

5. **운항 지연 및 연령/거리**
   - **지연 시간**: 90% 이상이 0분(정시 운항)인 Zero-inflated 분포를 가지며 지연 시간이 길어질수록 만족 비율이 완만히 하락합니다.
   - **연령**: 40~60대에서 만족 비율이 높고, 40세 미만 및 60세 이상에서는 불만족 비율이 우세합니다.
   - **데이터 드리프트**: Train과 Test 데이터 간 피처 분포(Covariate Shift)가 매우 일치하여 안정적인 검증 환경이 유지됩니다.

---

## 4. 베이스라인 모델 성능

- **로지스틱 회귀 (Logistic Regression)**
  - 단순 선형 벤치마크
  - 검증 ROC-AUC: $\approx 0.920$
- **LightGBM (Stratified 5-Fold CV)**
  - 비선형 패턴 및 고차 상호작용 학습
  - OOF(Out-of-Fold) 검증 ROC-AUC: $\approx 0.958$ (로지스틱 대비 약 $+0.038$p 향상)
  - **주요 기여 피처 (Feature Importance)**:
    1. `Online boarding`
    2. `Inflight wifi service`
    3. `Type of Travel`

---

## 5. 성능 향상을 위한 추천 전략 (Roadmap)

| 전략 영역 | 상세 내용 |
| :--- | :--- |
| **피처 엔지니어링** | - **0점 여부 플래그**: 특정 서비스 평점이 0점인지 나타내는 바이너리 피처 생성<br>- **지연 차이**: 순 지연시간 (`Arrival Delay` - `Departure Delay`) 및 총 지연시간 합<br>- **평점 집계 통계량**: 14개 서비스 평점의 평균(Mean), 표준편차(Std), 합계(Sum), 최소/최대값<br>- **비즈니스 결합 피처**: `Class`와 `Type of Travel`의 교차 범주 피처 |
| **다양한 트리 모델 도입** | - **CatBoost**: 범주형 피처 처리 및 대칭 트리 구조의 일반화 성능 우수<br>- **XGBoost**: LightGBM과 다른 트리 분할 방식 학습 |
| **딥러닝/신경망 모델** | - TabNet, MLP, ResNet 계열 정형 신경망 모델 구성 |
| **하이퍼파라미터 튜닝** | - Optuna를 통한 베이지안 최적화 (`learning_rate`, `num_leaves`, `min_child_samples`, `colsample_bytree`, `subsample` 등) |
| **앙상블 및 스태킹 (Stacking)** | - LGBM + CatBoost + XGBoost + MLP OOF 확률값 기반의 가중 블렌딩(Weighted Blending) 또는 Ridge/LGBM 메타 모델 스태킹 |
| **검증 원칙 (Protocol)** | - 퍼블릭 리더보드에 과적합되지 않도록 **로컬 Stratified 5-Fold OOF ROC-AUC**를 모델 채택 기준으로 유지 |
