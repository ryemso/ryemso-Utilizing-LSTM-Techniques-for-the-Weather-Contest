# Heat Demand Forecasting

기상 데이터와 지역난방 열수요를 결합해 **시간대별 열수요를 예측**한 시계열 프로젝트입니다.

## Problem

시간 단위 기상 변수와 과거 열수요를 이용해 미래 열수요를 예측했습니다.  
단순 모델 교체보다 **시퀀스 구성, 데이터 경계, 스케일링 방식이 검증 성능에 미치는 영향**을 함께 점검했습니다.

## Modeling

비교한 주요 구조:

- BiLSTM
- CNN-LSTM
- BiLSTM + Attention
- CNN + BiLSTM + Attention

주요 입력 변수는 기온, 풍향, 풍속, 강수량, 상대습도, 일사량, 체감온도, 시간, 요일 등입니다.

## What I Improved

### 1. Missing-value / preprocessing
원 데이터의 결측값과 기상 변수별 결측 패턴을 확인하고 여러 보간 방법을 비교했습니다.  
일사량처럼 결측 비율이 큰 변수는 제외 모델도 함께 검토했습니다.

### 2. Sequence continuity
Validation/Test 구간을 별도로 자르면 앞 시점 정보가 끊겨 예측 가능한 시퀀스가 누락될 수 있었습니다.  
Sliding Window와 앞 구간 패딩 방식을 사용해 경계 구간의 연속성을 보완했습니다.

### 3. Data leakage prevention
Scaler는 전체 데이터가 아니라 **훈련 데이터 기준으로만 fit**하고 Validation/Test에는 transform만 적용했습니다.

### 4. Model comparison
LSTM 계열 구조를 순차적으로 비교하면서 CNN의 국소 패턴 추출과 BiLSTM/Attention의 시간 정보 활용을 결합했습니다.

## Result

- 공모전 검증 과정에서 확인한 주요 RMSE: **21.7 → 17.2**
- 모델 구조뿐 아니라 전처리·시퀀스 생성·검증 방식까지 함께 수정

> 이 저장소의 결과 수치는 당시 프로젝트 검증 기록을 기준으로 정리했습니다.

## Repository Structure

```text
.
├── notebooks/
│   └── 01_heat_demand_modeling.ipynb
└── README.md
```

공개 노트북에는 데이터 로딩, Train-only scaling, 시퀀스 생성, BiLSTM 및 CNN+BiLSTM+Attention 모델 구성 코드가 포함되어 있습니다.  
데이터 파일은 저장소에 포함하지 않습니다.

## Tech Stack

**Python · Pandas · Scikit-learn · TensorFlow/Keras · LSTM · CNN · Attention · Time Series**

## Portfolio

[AI / Machine Learning Portfolio](https://kimsportpolio.netlify.app/?ver=ai)
