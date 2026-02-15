import os
import urllib.request
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# =====================================
# CONFIG
# =====================================

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# AUTO DOWNLOAD MODELS IF NOT EXISTS
# =====================================

RESNET_PATH = "best_resnet50.pth"
GROWTH_PATH = "growth_model.pkl"

# 🔴 PASTE YOUR DIRECT GOOGLE DRIVE LINKS BELOW
RESNET_URL = "import os
import urllib.request
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# =====================================
# CONFIG
# =====================================

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# AUTO DOWNLOAD MODELS IF NOT EXISTS
# =====================================

RESNET_PATH = "best_resnet50.pth"
GROWTH_PATH = "growth_model.pkl"

# 🔴 PASTE YOUR DIRECT GOOGLE DRIVE LINKS BELOW
RESNET_URL = "https://drive.google.com/file/d/169ZA_HNd8xWWAWI6Q6juv_bt2BVCEQc4/view?usp=sharing"
GROWTH_URL = "https://drive.google.com/file/d/1yoYk4CAV5B1Vs8b60wnAayONIpUETVFR/view?usp=sharing"

def download_file(url, path):
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"{path} downloaded successfully!")

if not os.path.exists(RESNET_PATH):
    download_file(RESNET_URL, RESNET_PATH)

if not os.path.exists(GROWTH_PATH):
    download_file(GROWTH_URL, GROWTH_PATH)

# =====================================
# LOAD GROWTH MODEL
# =====================================

pipeline = joblib.load(GROWTH_PATH)
print("✅ Growth Model Loaded")

# =====================================
# SUPPORT FUNCTIONS (Growth)
# =====================================

def estimate_yield(height):
    return round(height * 0.35, 2)

def irrigation_advice(temp, humidity, rainfall):
    if rainfall >= 20:
        return "No irrigation needed."
    if temp > 32 and humidity < 70:
        return "Irrigation recommended."
    return "Maintain monitoring."

def harvest_advice(height):
    if height < 250:
        return "Crop not ready."
    elif 250 <= height < 320:
        return "Approaching maturity."
    elif 320 <= height <= 400:
        return "Ready for harvest."
    else:
        return "Over-mature. Harvest immediately."

def fertilizer_recommendation(N, P, K, stage):

    stage_targets = {
        1: {"N": 40, "P": 60, "K": 60},
        2: {"N": 120, "P": 40, "K": 80},
        3: {"N": 100, "P": 40, "K": 120},
        4: {"N": 40, "P": 30, "K": 100}
    }

    targets = stage_targets.get(stage)

    nutrients = {
        "Nitrogen (N)": (N, targets["N"], 0.46, "Urea (46-0-0)"),
        "Phosphorus (P)": (P, targets["P"], 0.14, "Complete (14-14-14)"),
        "Potassium (K)": (K, targets["K"], 0.60, "MOP (0-0-60)")
    }

    results = []

    for name, (current, required, percent, fert_name) in nutrients.items():
        deficit = max(required - current, 0)

        if deficit > 0:
            fert_needed = round(deficit / percent, 2)
            bags = round(fert_needed / 50, 2)
            amount = f"{fert_needed} kg/ha"
            bags_text = f"{bags} bags/ha"
        else:
            amount = "Not Required"
            bags_text = "-"

        results.append({
            "nutrient": name,
            "current": current,
            "required": required,
            "fertilizer": fert_name,
            "amount": amount,
            "bags": bags_text
        })

    return results

# =====================================
# DISEASE MODEL
# =====================================

classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

disease_info = {
    "Healthy": "Malusog ang halaman. Panatilihin ang tamang patubig at balanseng abono.",
    "Mosaic": "Tanggalin ang infected na dahon at gumamit ng virus-free planting materials.",
    "RedRot": "Putulin at sunugin ang apektadong bahagi.",
    "Rust": "Mag-spray ng fungicide kung kinakailangan.",
    "Yellow": "Magdagdag ng sapat na nitrogen fertilizer."
}

model = models.resnet50(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(classes))
)

model.load_state_dict(torch.load(RESNET_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return classes[predicted.item()], round(confidence.item() * 100, 2)

# =====================================
# ROUTE
# =====================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    fertilizer_result = None
    prediction = None
    confidence = None
    info = None
    image_file = None

    if request.method == "POST" and "soil_pH" in request.form:

        soil_pH = float(request.form["soil_pH"])
        soil_type = request.form["soil_type"]
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall_irrigation"])
        stage = int(request.form["growth_stage"])

        input_data = pd.DataFrame([{
            "soil_pH": soil_pH,
            "soil_type": soil_type,
            "N": N,
            "P": P,
            "K": K,
            "temperature": temp,
            "humidity": humidity,
            "rainfall_irrigation": rainfall,
            "growth_stage": stage
        }])

        predicted_height = pipeline.predict(input_data)[0]

        result = {
            "height": round(predicted_height, 2),
            "yield": estimate_yield(predicted_height),
            "irrigation": irrigation_advice(temp, humidity, rainfall),
            "harvest": harvest_advice(predicted_height)
        }

        fertilizer_result = fertilizer_recommendation(N, P, K, stage)

    if request.method == "POST" and "file" in request.files:

        file = request.files["file"]

        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)
            info = disease_info[prediction]
            image_file = file.filename

    return render_template(
        "index.html",
        result=result,
        fertilizer=fertilizer_result,
        prediction=prediction,
        confidence=confidence,
        info=info,
        image_file=image_file
    )

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)"
GROWTH_URL = "PASTE_GROWTH_MODEL_DIRECT_LINK_HERE"

def download_file(url, path):
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"{path} downloaded successfully!")

if not os.path.exists(RESNET_PATH):
    download_file(RESNET_URL, RESNET_PATH)

if not os.path.exists(GROWTH_PATH):
    download_file(GROWTH_URL, GROWTH_PATH)

# =====================================
# LOAD GROWTH MODEL
# =====================================

pipeline = joblib.load(GROWTH_PATH)
print("✅ Growth Model Loaded")

# =====================================
# SUPPORT FUNCTIONS (Growth)
# =====================================

def estimate_yield(height):
    return round(height * 0.35, 2)

def irrigation_advice(temp, humidity, rainfall):
    if rainfall >= 20:
        return "No irrigation needed."
    if temp > 32 and humidity < 70:
        return "Irrigation recommended."
    return "Maintain monitoring."

def harvest_advice(height):
    if height < 250:
        return "Crop not ready."
    elif 250 <= height < 320:
        return "Approaching maturity."
    elif 320 <= height <= 400:
        return "Ready for harvest."
    else:
        return "Over-mature. Harvest immediately."

def fertilizer_recommendation(N, P, K, stage):

    stage_targets = {
        1: {"N": 40, "P": 60, "K": 60},
        2: {"N": 120, "P": 40, "K": 80},
        3: {"N": 100, "P": 40, "K": 120},
        4: {"N": 40, "P": 30, "K": 100}
    }

    targets = stage_targets.get(stage)

    nutrients = {
        "Nitrogen (N)": (N, targets["N"], 0.46, "Urea (46-0-0)"),
        "Phosphorus (P)": (P, targets["P"], 0.14, "Complete (14-14-14)"),
        "Potassium (K)": (K, targets["K"], 0.60, "MOP (0-0-60)")
    }

    results = []

    for name, (current, required, percent, fert_name) in nutrients.items():
        deficit = max(required - current, 0)

        if deficit > 0:
            fert_needed = round(deficit / percent, 2)
            bags = round(fert_needed / 50, 2)
            amount = f"{fert_needed} kg/ha"
            bags_text = f"{bags} bags/ha"
        else:
            amount = "Not Required"
            bags_text = "-"

        results.append({
            "nutrient": name,
            "current": current,
            "required": required,
            "fertilizer": fert_name,
            "amount": amount,
            "bags": bags_text
        })

    return results

# =====================================
# DISEASE MODEL
# =====================================

classes = ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']

disease_info = {
    "Healthy": "Malusog ang halaman. Panatilihin ang tamang patubig at balanseng abono.",
    "Mosaic": "Tanggalin ang infected na dahon at gumamit ng virus-free planting materials.",
    "RedRot": "Putulin at sunugin ang apektadong bahagi.",
    "Rust": "Mag-spray ng fungicide kung kinakailangan.",
    "Yellow": "Magdagdag ng sapat na nitrogen fertilizer."
}

model = models.resnet50(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(classes))
)

model.load_state_dict(torch.load(RESNET_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return classes[predicted.item()], round(confidence.item() * 100, 2)

# =====================================
# ROUTE
# =====================================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    fertilizer_result = None
    prediction = None
    confidence = None
    info = None
    image_file = None

    if request.method == "POST" and "soil_pH" in request.form:

        soil_pH = float(request.form["soil_pH"])
        soil_type = request.form["soil_type"]
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        rainfall = float(request.form["rainfall_irrigation"])
        stage = int(request.form["growth_stage"])

        input_data = pd.DataFrame([{
            "soil_pH": soil_pH,
            "soil_type": soil_type,
            "N": N,
            "P": P,
            "K": K,
            "temperature": temp,
            "humidity": humidity,
            "rainfall_irrigation": rainfall,
            "growth_stage": stage
        }])

        predicted_height = pipeline.predict(input_data)[0]

        result = {
            "height": round(predicted_height, 2),
            "yield": estimate_yield(predicted_height),
            "irrigation": irrigation_advice(temp, humidity, rainfall),
            "harvest": harvest_advice(predicted_height)
        }

        fertilizer_result = fertilizer_recommendation(N, P, K, stage)

    if request.method == "POST" and "file" in request.files:

        file = request.files["file"]

        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)
            info = disease_info[prediction]
            image_file = file.filename

    return render_template(
        "index.html",
        result=result,
        fertilizer=fertilizer_result,
        prediction=prediction,
        confidence=confidence,
        info=info,
        image_file=image_file
    )

# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
