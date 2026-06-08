import random
import pandas as pd

rows = []

for _ in range(5000):

    temp = round(random.uniform(16, 32), 1)
    humidity = round(random.uniform(25, 85), 1)

    temp_score = 100 - abs(temp - 22) * 4
    hum_score = 100 - abs(humidity - 50) * 2.2

    score = (
            temp_score * 0.78 +
            hum_score * 0.22
    )

    score += random.uniform(-5, 5)

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
    "pmv_sleepsync_dataset.csv",
    index=False
)

print(df.head())
print("\n생성 완료!")