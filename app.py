from flask import Flask, jsonify, request
from flask_cors import CORS
import serial
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification #HF used load the saved Tiny DistilBERT models

app = Flask(__name__)
CORS(app) #Start communication between react FE & Flask BE

SERIAL_PORT = "/dev/cu.usbmodemDCDA0C3CE6EC2" #USB port connection  to Arduino
BAUD_RATE = 9600

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) #open serial connection to nano
time.sleep(2)

SENSOR_LABELS = [
    "NORMAL",
    "HIGH_TEMP",
    "HIGH_HUMIDITY",
    "GAS_ANOMALY",
    "COMBINED_STRESS",
]

COMMAND_LABELS = [
    "EXPLAIN_PUMP",
    "CHECK_SOIL",
    "CHECK_TEMP",
    "CHECK_HUMIDITY",
    "CHECK_GAS",
    "GENERAL_STATUS",
]

# load tokenizer 
sensor_tokenizer = AutoTokenizer.from_pretrained(
    "saved_models/distilbert",
    use_fast=False
)

# load trained sensor SLM model
sensor_model = AutoModelForSequenceClassification.from_pretrained(
    "saved_models/distilbert"
)
#PUT in evaluation mode AKA:Predecting (dont change this)
sensor_model.eval()

# load tokenizer command
command_tokenizer = AutoTokenizer.from_pretrained(
    "saved_models/command_slm",
    use_fast=False
)
command_model = AutoModelForSequenceClassification.from_pretrained(
    "saved_models/command_slm"
)
command_model.eval()

#store latest reading
latest_data = {
    "soil": "0",
    "temp": "0",
    "humidity": "0",
    "gas": "0",
    "status": "WAITING",
    "pump": "OFF",
    "ai_message": "Waiting for real sensor readings.",
}


#convert num--> text input
def predict_sensor_status(soil, temp, humidity, gas):
    text = f"soil {soil} temperature {temp} humidity {humidity} gas {gas}"
  # tokenize the text
    inputs = sensor_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32,
    )

    with torch.no_grad():
        outputs = sensor_model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    return SENSOR_LABELS[pred_id]


def predict_command_intent(question):
    q = question.lower()

    # override for pump related questions ;/
    if "pump" in q or "water" in q or "irrigation" in q:
        return "EXPLAIN_PUMP"

    inputs = command_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32,
    )

    with torch.no_grad():
        outputs = command_model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()

    return COMMAND_LABELS[pred_id]


def make_sensor_message(status):
    if status == "COMBINED_STRESS":
        return "Multiple risk conditions detected. Irrigation and monitoring are recommended."
    if status == "HIGH_TEMP":
        return "High temperature detected. Monitor greenhouse conditions."
    if status == "HIGH_HUMIDITY":
        return "Humidity is above the safe range. Irrigation is not recommended."
    if status == "GAS_ANOMALY":
        return "Gas level is above normal range. Check air quality."
    if status == "NORMAL":
        return "Environmental conditions are stable."
    return "Waiting for valid sensor readings."


def answer_command(intent):
    soil = latest_data["soil"]
    temp = latest_data["temp"]
    humidity = latest_data["humidity"]
    gas = latest_data["gas"]
    pump = latest_data["pump"]
    status = latest_data["status"]

    if intent == "EXPLAIN_PUMP":
        if pump == "ON":
            return f"The command SLM detected a pump-related question. The pump is ON because the live soil moisture reading is {soil}, so irrigation is required."
        return f"The command SLM detected a pump-related question. The pump is OFF because the live soil moisture reading is {soil}, so irrigation is not required."

    if intent == "CHECK_SOIL":
        return f"The command SLM detected a soil-related question. The current soil moisture reading is {soil}, and the sensor SLM classified the condition as {status}."

    if intent == "CHECK_TEMP":
        return f"The command SLM detected a temperature-related question. The current temperature is {temp}°C, and the sensor SLM classified the condition as {status}."

    if intent == "CHECK_HUMIDITY":
        return f"The command SLM detected a humidity-related question. The current humidity is {humidity}%, and the sensor SLM classified the condition as {status}."

    if intent == "CHECK_GAS":
        return f"The command SLM detected an air-quality question. The current gas reading is {gas}, and the sensor SLM classified the condition as {status}."

    return f"The command SLM detected a general status request. Current condition: {status}. {latest_data['ai_message']}"


@app.route("/sensor-data")
def sensor_data():
    global latest_data

    try:
        line = arduino.readline().decode("utf-8").strip()

        if line:
            print("Arduino:", line)

            parsed = {}

            for part in line.split(","):
                if "=" in part:
                    key, value = part.split("=")
                    parsed[key] = value

            soil = parsed.get("soil", "0")
            temp = parsed.get("temp", "0")
            humidity = parsed.get("humidity", "0")
            gas = parsed.get("gas", "0")
            pump = parsed.get("pump", "OFF")

            sensor_status = predict_sensor_status(soil, temp, humidity, gas)

            latest_data = {
                "soil": soil,
                "temp": temp,
                "humidity": humidity,
                "gas": gas,
                "status": sensor_status,
                "pump": pump,
                "ai_message": make_sensor_message(sensor_status),
            }

    except Exception as e:
        print("Serial read error:", e)

    return jsonify(latest_data)


@app.route("/ask-command", methods=["POST"])
def ask_command():
    question = request.json.get("question", "")
    intent = predict_command_intent(question)
    answer = answer_command(intent)

    return jsonify({
        "intent": intent,
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)