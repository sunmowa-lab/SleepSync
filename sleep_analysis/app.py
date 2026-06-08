from flask import Flask, render_template, jsonify, request, session
import pandas as pd
import math
import os
import random
import joblib
import json
#import board
#import adafruit_dht
import requests
import RPi.GPIO as GPIO

import time
always_on_mode = False
lamp_enabled = False
dev_temp = None
dev_humidity = None

led_preview_until = 0
last_vibration_time = 0
last_vibration_detect = 0
last_motion_detect = 0
vibration_times = []
last_noise_time = 0





app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")

GPIO.setmode(GPIO.BCM)

RED_PIN = 23
GREEN_PIN = 24
BLUE_PIN = 25

GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)



LAST_SLEEP_FILE = "last_sleep_report.json"
DATA_FILE = 'sleep_log_score.csv'
DEV_PASSWORD = os.getenv("DEV_PASSWORD", "12345678")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "sleepsync_model.pkl"
)

sleep_model = joblib.load(MODEL_PATH)

#dht = adafruit_dht.DHT22(board.D4)

# 시스템 상태 변수 (서버 메모리)
system_status = {
    "temp": 22.0,
    "humidity": 50.0,
    "toss_turn": 0,
    "noise": 0,
    "motion": 0,
    "sleeping": False
}
sleep_session = {
    "temp_sum": 0,
    "humidity_sum": 0,
    "sample_count": 0,

    "toss_count": 0,
    "noise_count": 0,
    "motion_count": 0
}

if os.path.exists(LAST_SLEEP_FILE):

    with open(
        LAST_SLEEP_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        last_sleep_report = json.load(f)

else:

    last_sleep_report = {
        "avg_temp": 0,
        "avg_humidity": 0,
        "toss_count": 0,
        "noise_count": 0,
        "motion_count": 0
    }

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROFILE_FILE = os.path.join(
    BASE_DIR,
    "user_profile.json"
)

print("PROFILE FILE =", PROFILE_FILE)

if os.path.exists(PROFILE_FILE):

    with open(
        PROFILE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        user_profile = json.load(f)

else:

    user_profile = {
        "bias": 0.0
    }

    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            user_profile,
            f,
            ensure_ascii=False,
            indent=4
        )

def calculate_pmv(temp, humidity):
    p_sat = 0.61078 * math.exp((17.27 * temp) / (temp + 237.3))
    p_v = (humidity / 100.0) * p_sat
    internal_load = 0.15 * temp + 0.003 * p_v - 3.65
    return max(-3.0, min(3.0, internal_load))

def set_led_color(color_name):
    
    print(f"LED CHANGE: {color_name}")

    GPIO.output(RED_PIN, False)
    GPIO.output(GREEN_PIN, False)
    GPIO.output(BLUE_PIN, False)

    if color_name == "blue":
        GPIO.output(BLUE_PIN, True)

    elif color_name == "green":
        GPIO.output(GREEN_PIN, True)

    elif color_name == "yellow":
        GPIO.output(RED_PIN, True)
        GPIO.output(GREEN_PIN, True)

    elif color_name == "red":
        GPIO.output(RED_PIN, True)


def get_env_evaluation(pmv):
    abs_pmv = abs(pmv)
    if abs_pmv <= 0.5:
        
        return "매우 쾌적", "#03A9F4", "현재 침실 환경은 수면에 매우 적합합니다. 파란색 계열 조명이 유지됩니다."

    elif abs_pmv <= 1.5:
      
        return "수면 주의", "#FF9800", "온습도가 다소 불균형합니다. 실내 조절기 확인을 권장하는 노란색 조명이 점등됩니다."

    else:
       
        return "수면 불량", "#E91E63", "수면 환경 개선이 필요합니다. 환경 조절 솔루션 유도가 필요한 빨간색 조명이 점등됩니다."

def predict_sleep_score(temp, humidity):

    base_score = sleep_model.predict(
        pd.DataFrame(
            [[temp, humidity]],
            columns=[
                "Temperature",
                "Humidity"
            ]
        )
    )[0]

    base_score = float(base_score)

    temp_distance = abs(
    temp - user_profile["preferred_temp"]
)

    humidity_distance = abs(
        humidity - user_profile["preferred_humidity"]
    )

    temp_bonus = max(
        0,
        100 - temp_distance * 8
    )

    humidity_bonus = max(
        0,
        100 - humidity_distance * 2
    )

    personalized_score = (
        base_score * 0.7
        + temp_bonus * 0.2
        + humidity_bonus * 0.1
    )
    personalized_score += user_profile["bias"]

    personalized_score = max(
        0,
        min(
            100,
            personalized_score
        )
    )

    return {
    "base_score": round(base_score, 1),
    "personalized_score": round(personalized_score, 1)
}

def is_dev_mode():
    return session.get("is_dev_mode", False)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json()

    temp = float(data["temp"])
    humidity = float(data["humidity"])

    result = predict_sleep_score(
        temp,
        humidity
    )

    return jsonify(result)


@app.route('/api/toggle-lamp', methods=['POST'])
def toggle_lamp():

    global lamp_enabled

    lamp_enabled = not lamp_enabled

    return jsonify({
        "enabled": lamp_enabled
    })
# 1. 개발자 모드 인증 API
@app.route('/api/auth', methods=['POST'])
def auth_dev():
    data = request.get_json(silent=True) or {}

    if data.get("password") == DEV_PASSWORD:
        session["is_dev_mode"] = True
        return jsonify({"success": True, "message": "개발자 모드가 활성화되었습니다."})

    return jsonify({"success": False, "message": "비밀번호가 틀렸습니다."}), 401


# 개발자 모드 종료 API
@app.route('/api/logout-dev', methods=['POST'])
def logout_dev():

    global dev_temp
    global dev_humidity

    session["is_dev_mode"] = False

    dev_temp = None
    dev_humidity = None

    return jsonify({
        "success": True,
        "message": "개발자 모드가 종료되었습니다."
    })
@app.route('/api/reset-profile', methods=['POST'])
def reset_profile():
    global last_sleep_report
    global dev_temp
    global dev_humidity
    global user_profile

    last_sleep_report = {
        "avg_temp": 0,
        "avg_humidity": 0,
        "toss_count": 0,
        "noise_count": 0,
        "motion_count": 0
    }

    with open(
        LAST_SLEEP_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            last_sleep_report,
            f,
            ensure_ascii=False,
            indent=4
        )

    user_profile = {
    "preferred_temp": 22.0,
    "preferred_humidity": 50.0,
    "feedback_count": 0,
    "bias": 0.0
    }

    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            user_profile,
            f,
            ensure_ascii=False,
            indent=4
        )

    df = pd.DataFrame(
        columns=[
            'Temp',
            'Humidity',
            'PMV',
            'TossTurn',
            'Noise',
            'SleptWell'
        ]
    )

    df.to_csv(
        DATA_FILE,
        index=False
    )
    dev_temp = None
    dev_humidity = None
    return jsonify({
        "success": True,
        "message": "AI 학습 데이터와 수면 기록이 초기화되었습니다."
    })

# 2. 실시간 상태 동기화 API
@app.route('/api/status', methods=['GET', 'POST'])
def handle_status():
    # 개발자 모드일 때만 POST로 들어온 온도/습도 조작 허용
    global last_vibration_time
    global led_preview_until
    global always_on_mode
    global vibration_times
    global last_noise_time
    global sleep_session
    global last_vibration_detect
    global lamp_enabled
    global dev_temp
    global dev_humidity
    global last_motion_detect



    if request.method == 'POST':
        if not is_dev_mode():
            return jsonify({"success": False, "message": "권한이 없습니다."}), 403

        data = request.get_json(silent=True) or {}
        if "temp" in data:
            dev_temp = float(data["temp"])

        if "humidity" in data:
            dev_humidity = float(data["humidity"])

    try:
        sensor = requests.get(
            "http://127.0.0.1:5001",
            timeout=2
        ).json()

        # 개발자 모드가 아닐 때만 실제 센서값 사용
        if sensor["temp"] is not None:
            system_status["temp"] = sensor["temp"]

        if sensor["humidity"] is not None:
            system_status["humidity"] = sensor["humidity"]
        # 계산에 사용할 온습도 결정

        calc_temp = system_status["temp"]
        calc_humidity = system_status["humidity"]

        if is_dev_mode():

            if dev_temp is not None:
                calc_temp = dev_temp

            if dev_humidity is not None:
                calc_humidity = dev_humidity    


        system_status["noise"] = sensor["touch"]
        system_status["toss_turn"] = sensor["motion"]
        system_status["motion"] = sensor["vibration"]




        print("STATUS CALLED")
        print("sleeping =", system_status["sleeping"])

        if system_status["sleeping"]:

            sleep_session["temp_sum"] += system_status["temp"]
            sleep_session["humidity_sum"] += system_status["humidity"]

            sleep_session["sample_count"] += 1

            global last_noise_time

            if sensor["touch"] == 1:

                sleep_session["noise_count"] += 1

                last_noise_time = time.time()
            if sensor["motion"] == 1:
                now = time.time()
                if now - last_motion_detect > 5:
                    sleep_session["toss_count"] += 1
                    last_motion_detect = now    

        # ===== SW420 처리 =====

        if sensor["vibration"] == 1:
            last_vibration_detect = time.time()

        if time.time() - last_vibration_detect < 0.5:

            now = time.time()

            if now - last_vibration_time > 0.5:

                last_vibration_time = now

                led_preview_until = now + 7

                vibration_times.append(now)

                vibration_times[:] = [
                    t for t in vibration_times
                    if now - t <= 15
                ]

                if len([
                    t for t in vibration_times
                    if now - t <= 10
                ]) >= 3:

                    sleep_session = {
                        "temp_sum": 0,
                        "humidity_sum": 0,
                        "sample_count": 0,
                        "toss_count": 0,
                        "noise_count": 0,
                        "motion_count": 0
                    }

                    system_status["sleeping"] = True

                    vibration_times.clear()

                    # =========================
                    # 현재 수면점수 색상으로 3회 깜빡임
                    # =========================

                    score = predict_sleep_score(
                        calc_temp,
                        calc_humidity
                    )["personalized_score"]

                    if score >= 85:
                        blink_color = "blue"

                    elif score >= 70:
                        blink_color = "green"

                    elif score >= 50:
                        blink_color = "yellow"

                    else:
                        blink_color = "red"

                    for _ in range(3):

                        set_led_color(blink_color)
                        time.sleep(0.2)

                        GPIO.output(RED_PIN, False)
                        GPIO.output(GREEN_PIN, False)
                        GPIO.output(BLUE_PIN, False)
                        time.sleep(0.2)

                    print("REAL START SLEEP (3 TOUCH)")
                    print(system_status)

             
        # ===== LED 제어 =====

        now = time.time()

        if lamp_enabled:

            score = predict_sleep_score(
                calc_temp,
                calc_humidity
            )["personalized_score"]

            if score >= 85:
                set_led_color("blue")

            elif score >= 70:
                set_led_color("green")

            elif score >= 50:
                set_led_color("yellow")

            else:
                set_led_color("red")

        elif now < led_preview_until:
            
            score = predict_sleep_score(
                calc_temp,
                calc_humidity
            )["personalized_score"]

            if score >= 85:
                set_led_color("blue")

            elif score >= 70:
                set_led_color("green")

            elif score >= 50:
                set_led_color("yellow")

            else:
                set_led_color("red")

        else:

            GPIO.output(RED_PIN, False)
            GPIO.output(GREEN_PIN, False)
            GPIO.output(BLUE_PIN, False)            
              

    except Exception as e:
        print("센서 연결 실패:", e)
    pmv_val = calculate_pmv(
        calc_temp,
        calc_humidity
    )

    status_label, color_code, ai_analysis = get_env_evaluation(
        pmv_val
    )
    
    

    return jsonify({
    "is_dev_mode": is_dev_mode(),
    "temp": calc_temp,
    "humidity": calc_humidity,
    "pmv": round(pmv_val, 2),
    "status_label": status_label,
    "color_code": color_code,
    "ai_analysis": ai_analysis,
    "toss_turn": system_status["toss_turn"],
    "noise": system_status["noise"],
    "motion": system_status["motion"],
    "sleeping": system_status["sleeping"]

})


# 3. [개발자용] 다음 날로 강제 점프 API
@app.route('/api/lamp-status')
def lamp_status():

    return jsonify({
        "enabled": lamp_enabled
    })

@app.route('/api/dev/next-day', methods=['POST'])
def dev_next_day():
    if not is_dev_mode():
        return jsonify({"success": False, "message": "권한이 없습니다."}), 403

    pmv_val = calculate_pmv(system_status["temp"], system_status["humidity"])
    base_error = int(abs(pmv_val) * 10)
    system_status["toss_turn"] = random.randint(5 + base_error, 25 + base_error)
    system_status["noise"] = random.randint(2 + base_error, 12 + base_error)

    return jsonify({"success": True, "message": "시간 가속 완료! 밤사이 데이터가 생성되었습니다."})

@app.route('/api/start-sleep', methods=['POST'])
def start_sleep():

    global sleep_session

    sleep_session = {
        "temp_sum": 0,
        "humidity_sum": 0,
        "sample_count": 0,

        "toss_count": 0,
        "noise_count": 0,
        "motion_count": 0
    }

    system_status["sleeping"] = True

    print("REAL START SLEEP")
    print(system_status)

    return jsonify({
        "success": True,
        "message": "수면 측정이 시작되었습니다."
    })
@app.route('/api/set-led', methods=['POST'])
def set_led_api():

    return jsonify({"success": True})

# 4. [일반 유저용] 기상 후 수면 점수 기록 API
@app.route('/api/save-score', methods=['POST'])
def save_score():
    print("SAVE SCORE CALLED")

    data = request.get_json(silent=True) or {}

    score = float(data.get("score", 50))
    learn_temp = system_status["temp"]
    learn_humidity = system_status["humidity"]

    if is_dev_mode():

        if dev_temp is not None:
            learn_temp = dev_temp

        if dev_humidity is not None:
            learn_humidity = dev_humidity

    
    print("===== PERSONALIZATION DEBUG =====")

    print(
        "temp =",
        system_status["temp"]
    )

    print(
        "humidity =",
        system_status["humidity"]
    )

    print(
        "user score =",
        score
    )

   
    print(user_profile)
    prediction = predict_sleep_score(
        learn_temp,
        learn_humidity
    )

    predicted_score = prediction["personalized_score"]

    error = score - predicted_score

    learning_rate = 0.05

    user_profile["bias"] += (
        error * learning_rate
    )
    preference_rate = 0.03

    if error > 10:

        user_profile["preferred_temp"] += (
            learn_temp
            - user_profile["preferred_temp"]
        ) * preference_rate

        user_profile["preferred_humidity"] += (
            learn_humidity
            - user_profile["preferred_humidity"]
        ) * preference_rate
        
    elif error < -10:

        user_profile["preferred_temp"] -= (
            learn_temp
            - user_profile["preferred_temp"]
        ) * preference_rate

        user_profile["preferred_humidity"] -= (
            learn_humidity
            - user_profile["preferred_humidity"]
        ) * preference_rate    

    user_profile["feedback_count"] += 1

    print("predicted =", predicted_score)
    print("actual =", score)
    print("error =", error)
    print("updated bias =", user_profile["bias"])


    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            user_profile,
            f,
            ensure_ascii=False,
            indent=4
        )

    if not os.path.exists(DATA_FILE):

        df = pd.DataFrame(
            columns=[
                'Temp',
                'Humidity',
                'PMV',
                'TossTurn',
                'Noise',
                'SleptWell'
            ]
        )

    else:

        df = pd.read_csv(DATA_FILE)

   
    
    pmv_val = calculate_pmv(
        learn_temp,
        learn_humidity
    )

    new_row = pd.DataFrame({
    'Temp': [learn_temp],
    'Humidity': [learn_humidity],
    'PMV': [pmv_val],
    'TossTurn': [system_status["toss_turn"]],
    'Noise': [system_status["noise"]],
    'SleptWell': [score]
    })

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    df.to_csv(
        DATA_FILE,
        index=False
    )
   
    print("sample_count =", sleep_session["sample_count"])
    print("temp_sum =", sleep_session["temp_sum"])
    print("humidity_sum =", sleep_session["humidity_sum"])

    global last_sleep_report

    if sleep_session["sample_count"] > 0:

        last_sleep_report = {

            "avg_temp": round(
                sleep_session["temp_sum"]
                / sleep_session["sample_count"],
                1
            ),

            "avg_humidity": round(
                sleep_session["humidity_sum"]
                / sleep_session["sample_count"],
                1
            ),

            "toss_count":
                sleep_session["toss_count"],

            "noise_count":
                sleep_session["noise_count"],

            "motion_count":
                sleep_session["motion_count"]
        }

        print("===== LAST SLEEP REPORT =====")
        print(last_sleep_report)
        print("SAVE FILE =", LAST_SLEEP_FILE)
        
        with open(
            LAST_SLEEP_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                last_sleep_report,
                f,
                ensure_ascii=False,
                indent=4
            )

    system_status["toss_turn"] = 0
    system_status["noise"] = 0
    system_status["sleeping"] = False

    return jsonify({
        "success": True,
        "message": "오늘의 수면 점수가 안전하게 기록되었습니다!"
    })


print(
    predict_sleep_score(
        22,
        50
    )
)
@app.route('/api/last-sleep')
def last_sleep():

    return jsonify(last_sleep_report)

@app.route('/api/sleep-history')
def sleep_history():

    if not os.path.exists(DATA_FILE):
        return jsonify([])

    df = pd.read_csv(DATA_FILE)

    if len(df) == 0:
        return jsonify([])

    scores = df["SleptWell"].tail(7).tolist()

    return jsonify(scores)

  
if __name__ == "__main__":
    import logging

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(
    host='0.0.0.0',
    port=5000,
    debug=False
    )   
