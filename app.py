from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from ultralytics import YOLO
import os
import shutil
import glob

app = Flask(__name__)
app.secret_key = "secret_key"

# # Directories for uploads and predictions
# UPLOAD_FOLDER = 'static/uploads'
# PREDICTIONS_FOLDER = 'static/predictions'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Get the directory where the main.py file is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the main.py file's location
UPLOAD_FOLDER = os.path.join(base_dir, 'static', 'uploads')
PREDICTIONS_FOLDER = os.path.join(base_dir, 'static', 'predictions')

# Create the directories if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Load the YOLO model
model = YOLO("yolo11m.pt")

@app.route('/')
def index():
    """Render the homepage."""
    # Check for any file in the predictions folder
    predicted_files = glob.glob(os.path.join(PREDICTIONS_FOLDER, "*"))
    predicted_image = None
    if predicted_files:
        predicted_image = os.path.basename(predicted_files[0])  # Get the first image found

    return render_template('index.html', predicted_image=predicted_image)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform YOLO predictions."""
    # Clear previous YOLO outputs
    runs_folder = os.path.join("runs", "detect", "predict")
    if os.path.exists(runs_folder):
        shutil.rmtree(runs_folder)

    # Clear previous predictions in the PREDICTIONS_FOLDER
    shutil.rmtree(PREDICTIONS_FOLDER)
    os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

    if 'image' not in request.files:
        flash("No file part in the request.")
        return redirect(request.url)

    file = request.files['image']  # Get the uploaded file

    if file.filename == '':
        flash("No file selected for uploading.")
        return redirect(request.url)

    if file:
        # Save the uploaded file to the UPLOAD_FOLDER
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Run YOLO predictions
        model.predict(source=file_path, save=True, imgsz=320, save_txt=True)

        # Path to YOLO's output
        predicted_image_path = os.path.join("runs", "detect", "predict", os.path.basename(file.filename))
        label_file_path = os.path.join("runs", "detect", "predict", "labels", os.path.splitext(os.path.basename(file.filename))[0] + ".txt")

        # Copy the predicted image to the PREDICTIONS_FOLDER
        if os.path.exists(predicted_image_path):
            shutil.copy(predicted_image_path, os.path.join(PREDICTIONS_FOLDER, os.path.basename(file.filename)))

        # Count detected objects
        count = 0
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as label_file:
                count = sum(1 for line in label_file if line.startswith("67") or line.startswith("65"))

        # Redirect to home with results
        return render_template(
            'index.html',
            original_image=file.filename,
            predicted_image=os.path.join(PREDICTIONS_FOLDER, os.path.basename(file.filename)),
            count=count
        )

    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded file."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/predictions')
def predicted_file():
    """Serve the predicted file (if it exists)."""
    predicted_files = glob.glob(os.path.join(PREDICTIONS_FOLDER, "*"))
    if predicted_files:
        return send_from_directory(PREDICTIONS_FOLDER, os.path.basename(predicted_files[0]))
    else:
        return "No predicted file found.", 404
    
if __name__ == '__main__':
    app.run()

# if __name__ == '__main__':
#     app.run(debug=True)





