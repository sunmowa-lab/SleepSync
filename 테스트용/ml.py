import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

# ==================================================
# 데이터 로드
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "pmv_sleepsync_dataset.csv"
)

df = pd.read_csv(csv_path)

print("\n===== 데이터 확인 =====")
print(df.head())

# 입력값(X)
X = df[["Temperature", "Humidity"]]

# 정답(y)
y = df["SleepScore"]

# 학습용 / 테스트용 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 모델 생성
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=8,
    random_state=42
)

# 학습
model.fit(X_train, y_train)

# 성능 확인
pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    pred
)

print(f"평균 오차(MAE): {mae:.2f}")

# 모델 저장
joblib.dump(
    model,
    "sleepsync_model.pkl"
)

print("sleepsync_model.pkl 저장 완료")

# 테스트
sample = pd.DataFrame(
    [[22, 50]],
    columns=["Temperature", "Humidity"]
)

score = model.predict(sample)[0]
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 성능 확인
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\n===== 모델 평가 결과 =====")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# 변수 중요도
importance = pd.DataFrame({
    "변수": ["온도", "습도"],
    "중요도": model.feature_importances_
})

print("\n===== 변수 중요도 =====")
print(importance)
# ==================================================
# 시각화
# ==================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ==================================================
# 1. PMV 적용 수면 점수 분포
# ==================================================

plt.figure(figsize=(12, 6))

plt.hist(
    df["SleepScore"],
    bins=20
)

plt.title("PMV 적용 수면 점수 분포")
plt.xlabel("수면 점수")
plt.ylabel("데이터 개수")

plt.grid(True)

plt.show()


# ==================================================
# 2. 100개 랜덤 환경 예측 결과
# ==================================================

random_scores = []

for _ in range(100):

    temp = random.uniform(16, 32)
    humidity = random.uniform(25, 85)

    sample = pd.DataFrame(
        [[temp, humidity]],
        columns=[
            "Temperature",
            "Humidity"
        ]
    )

    score = model.predict(sample)[0]

    random_scores.append(score)

plt.figure(figsize=(12, 6))

plt.plot(random_scores)

plt.title(
    "100개의 임의 환경에 대한 수면 점수 예측 결과"
)

plt.xlabel("샘플 번호")
plt.ylabel("예측 수면 점수")

plt.grid(True)

plt.show()


# ==================================================
# 3. 랜덤 포레스트 학습 경과
# ==================================================

tree_range = range(1, 101)

r2_scores = []

for n in tree_range:

    temp_model = RandomForestRegressor(
        n_estimators=n,
        max_depth=8,
        random_state=42
    )

    temp_model.fit(
        X_train,
        y_train
    )

    temp_pred = temp_model.predict(
        X_test
    )

    r2_scores.append(
        r2_score(
            y_test,
            temp_pred
        )
    )

plt.figure(figsize=(12, 6))

plt.plot(
    tree_range,
    r2_scores
)

plt.title(
    "랜덤 포레스트 학습 경과"
)

plt.xlabel("트리 개수")
plt.ylabel("R² 점수")

plt.grid(True)

plt.show()


# ==================================================
# 4. 실제값 vs 예측값 비교
# ==================================================

plt.figure(figsize=(10, 8))

plt.scatter(
    y_test,
    pred,
    alpha=0.7
)

min_score = min(
    y_test.min(),
    pred.min()
)

max_score = max(
    y_test.max(),
    pred.max()
)

plt.plot(
    [min_score, max_score],
    [min_score, max_score],
    "r--",
    linewidth=3
)

plt.title(
    "실제 수면 점수와 예측 수면 점수 비교"
)

plt.xlabel("실제 수면 점수")
plt.ylabel("예측 수면 점수")

plt.grid(True)

plt.show()