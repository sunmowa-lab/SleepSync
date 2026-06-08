import pandas as pd
import joblib

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

print(f"22℃ / 50% 예측 점수 = {score:.1f}")