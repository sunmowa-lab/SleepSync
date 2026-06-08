import random
import pandas as pd

rows = []

for _ in range(5000):
    temp = round(random.uniform(16, 32), 1)
    humidity = round(random.uniform(25, 85), 1)

    ideal_temp = random.uniform(18, 24)

    temp_penalty = abs(temp - ideal_temp) * 8

    humidity_penalty = abs(humidity - 50) * 0.05

    score = 100 - temp_penalty - humidity_penalty

    # 측정 오차
    score += random.uniform(-7, 7)

    score = max(0, min(100, score))

    rows.append([
        temp,
        humidity,
        round(score, 1)
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "Temperature",
        "Humidity",
        "SleepScore"
    ]
)

df.to_csv(
    "sleepsync_dataset.csv",
    index=False
)

print(df.head())
print("\n생성 완료")