import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_PATH = "snake_model.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print("Warning: 'snake_model.h5' not found.")

# --- STUNNING FRONTEND HTML & CSS ---
# (Saved inside a Python string variable so you don't need a separate HTML file!)
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Snake Species Classifier - AI Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            color: #ecf0f1;
            /* ATTRACTIVE RELATED BACKGROUND FROM A FREE RELIABLE WEB URL */
            background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.85)), 
                             url('https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?q=80&w=1000');
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-color: #0f140f;
        }

        /* GLASSMORPHISM UI CARD */
        .card {
            max-width: 480px;
            width: 90%;
            background: rgba(255, 255, 255, 0.06);
            padding: 40px;
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }

        h2 {
            margin-top: 0;
            font-size: 26px;
            font-weight: 700;
            color: #2ecc71;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        .subtitle {
            color: #bdc3c7;
            margin-bottom: 30px;
            font-size: 15px;
        }

        /* DASHED UPLOAD BOX */
        .upload-box {
            border: 2px dashed rgba(46, 204, 113, 0.4);
            padding: 35px;
            border-radius: 16px;
            cursor: pointer;
            background: rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }

        .upload-box:hover {
            border-color: #2ecc71;
            background: rgba(46, 204, 113, 0.05);
        }

        #uploadText {
            color: #bdc3c7;
            font-size: 15px;
            margin: 0;
        }

        #preview {
            max-width: 100%;
            max-height: 220px;
            display: none;
            margin: 0 auto;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        }

        /* GLOWING BUTTON */
        .btn {
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white;
            border: none;
            padding: 16px;
            font-size: 16px;
            border-radius: 12px;
            cursor: pointer;
            width: 100%;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
            transition: all 0.2s ease;
        }

        .btn:hover {
            transform: scale(1.01);
            box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4);
        }

        .btn:disabled {
            background: #7f8c8d;
            box-shadow: none;
            cursor: not-allowed;
            opacity: 0.5;
        }

        /* RESPONSE BOXES */
        #result {
            margin-top: 25px;
            padding: 18px;
            border-radius: 12px;
            display: none;
            font-size: 20px;
            font-weight: bold;
            letter-spacing: 0.5px;
        }

        .venomous {
            background-color: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
            border: 1px solid rgba(231, 76, 60, 0.5);
        }

        .non-venomous {
            background-color: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
            border: 1px solid rgba(46, 204, 113, 0.5);
        }

        #resConf {
            font-size: 14px;
            font-weight: normal;
            color: #bdc3c7;
            margin-top: 5px;
        }
    </style>
</head>
<body>

<div class="card">
    <h2>Snake Classifier AI</h2>
    <p class="subtitle">Indian Species Venomous & Non-Venomous Verification Matrix</p>

    <div class="upload-box" onclick="document.getElementById('fileInput').click()">
        <p id="uploadText"><strong>Click to select a photo</strong><br><span style="font-size: 12px; color: #7f8c8d;">PNG, JPG, or JPEG</span></p>
        <input type="file" id="fileInput" accept="image/*" style="display: none;">
        <img id="preview">
    </div>

    <button id="classifyBtn" class="btn" disabled onclick="sendToAI()">Classify Image</button>

    <div id="result">
        <div id="resLabel"></div>
        <div id="resConf"></div>
    </div>
</div>

<script>
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const uploadText = document.getElementById('uploadText');
    const classifyBtn = document.getElementById('classifyBtn');
    const resultDiv = document.getElementById('result');

    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
                uploadText.style.display = 'none';
                classifyBtn.disabled = false;
                resultDiv.style.display = 'none';
            }
            reader.readAsDataURL(file);
        }
    });

    function sendToAI() {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        classifyBtn.disabled = true;
        classifyBtn.innerText = "Analyzing Matrix Arrays...";

        fetch('/predict', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            classifyBtn.disabled = false;
            classifyBtn.innerText = "Classify Image";

            if (data.error) { alert(data.error); return; }

            resultDiv.className = data.label === "Venomous" ? "venomous" : "non-venomous";
            document.getElementById('resLabel').innerText = "⚠️ RESULT: " + data.label;
            document.getElementById('resConf').innerText = "AI System Confidence: " + data.confidence;
            resultDiv.style.display = 'block';
        })
        .catch(err => {
            classifyBtn.disabled = false;
            classifyBtn.innerText = "Classify Image";
            alert("Connection error to core engine.");
        });
    }
</script>

</body>
</html>
"""

# --- FLASK WEBPAGE ROUTE ---
@app.route('/')
def home():
    # Renders the HTML string directly without needing a separate file/folder!
    return render_template_string(HTML_PAGE)

# --- ENGINE PREDICTION ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'snake_model.h5 weights file missing.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No data payload found.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if file:
        temp_path = "temp_check.jpg"
        file.save(temp_path)
        try:
            # Process tensor arrays
            img = image.load_img(temp_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            raw_score = float(prediction[0][0])

            if raw_score > 0.5:
                label = "Venomous"
                confidence = raw_score * 100
            else:
                label = "Non-Venomous"
                confidence = (1 - raw_score) * 100

            os.remove(temp_path)
            return jsonify({'label': label, 'confidence': f"{confidence:.2f}%"})
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)