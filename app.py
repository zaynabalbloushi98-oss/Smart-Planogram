
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import Counter, defaultdict
import os
import cv2
import uuid

app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

model = YOLO(MODEL_PATH)


# --------------------------------------------------
# EXPECTED VISUAL PLANOGRAM
# --------------------------------------------------

PLANOGRAM = {

    "pepsi_zero_sugar": {
        "display_name": "Pepsi Zero Sugar",
        "expected_quantity": 4,
        "expected_position": "left"
    },

    "fanta_orange": {
        "display_name": "Fanta Orange",
        "expected_quantity": 4,
        "expected_position": "center"
    },

    "redbull": {
        "display_name": "Red Bull",
        "expected_quantity": 4,
        "expected_position": "right"
    }
}


# --------------------------------------------------
# POSITION FUNCTION
# --------------------------------------------------

def get_zone(center_x, image_width):

    relative_x = center_x / image_width

    if relative_x < (1 / 3):
        return "left"

    elif relative_x < (2 / 3):
        return "center"

    else:
        return "right"


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# ANALYZE IMAGE
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    # Check if an image was uploaded
    if "image" not in request.files:
        return "No image uploaded", 400

    file = request.files["image"]

    if file.filename == "":
        return "No image selected", 400


    # Analysis mode:
    # full = quantity + position
    # detection = quantity only
    analysis_mode = request.form.get(
        "analysis_mode",
        "full"
    )


    # Create safe unique filename
    safe_name = secure_filename(file.filename)

    unique_name = (
        f"{uuid.uuid4().hex}_{safe_name}"
    )

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    file.save(upload_path)


    # --------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------

    predictions = model.predict(
        source=upload_path,
        conf=0.50,
        iou=0.50,
        verbose=False
    )

    result = predictions[0]

    image_width = result.orig_shape[1]

    counts = Counter()
    positions = defaultdict(list)


    # Process every detected product
    for box in result.boxes:

        class_id = int(box.cls[0])

        class_name = result.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2

        zone = get_zone(
            center_x,
            image_width
        )

        counts[class_name] += 1

        positions[class_name].append(zone)


    # --------------------------------------------------
    # PLANOGRAM COMPARISON
    # --------------------------------------------------

    comparison = []

    total_expected = sum(
        item["expected_quantity"]
        for item in PLANOGRAM.values()
    )

    total_quantity_error = 0

    correct_position_total = 0


    for product, info in PLANOGRAM.items():

        expected_qty = info["expected_quantity"]
        expected_zone = info["expected_position"]

        detected_qty = counts.get(product, 0)
        zones = positions.get(product, [])


        # --------------------------
        # QUANTITY STATUS
        # --------------------------

        if detected_qty == expected_qty:

            quantity_status = "Correct"

        elif detected_qty < expected_qty:

            quantity_status = (
                f"Missing {expected_qty - detected_qty}"
            )

        else:

            quantity_status = (
                f"Extra {detected_qty - expected_qty}"
            )


        # Missing AND Extra affect quantity compliance
        total_quantity_error += abs(
            expected_qty - detected_qty
        )


        # --------------------------
        # POSITION ANALYSIS
        # --------------------------

        if analysis_mode == "full":

            correct_in_zone = min(
                zones.count(expected_zone),
                expected_qty
            )

            correct_position_total += correct_in_zone


            if detected_qty == 0:

                position_status = "Product Missing"

            else:

                misplaced = sum(
                    1
                    for zone in zones
                    if zone != expected_zone
                )

                if misplaced == 0:

                    position_status = "Correct"

                else:

                    position_status = (
                        f"Misplaced {misplaced}"
                    )


            detected_position_text = (
                ", ".join(zones)
                if zones
                else "-"
            )


        else:

            position_status = "Not Evaluated"
            detected_position_text = "N/A"


        comparison.append({

            "product":
                info["display_name"],

            "expected":
                expected_qty,

            "detected":
                detected_qty,

            "quantity_status":
                quantity_status,

            "expected_position":
                expected_zone.title(),

            "detected_positions":
                detected_position_text,

            "position_status":
                position_status
        })


    # --------------------------------------------------
    # QUANTITY COMPLIANCE
    # --------------------------------------------------

    quantity_score = max(
        0,
        (
            1 -
            total_quantity_error / total_expected
        ) * 100
    )


    # --------------------------------------------------
    # POSITION + OVERALL COMPLIANCE
    # --------------------------------------------------

    if analysis_mode == "full":

        position_score = (
            correct_position_total /
            total_expected
        ) * 100

        overall_score = (
            quantity_score +
            position_score
        ) / 2

    else:

        position_score = None
        overall_score = quantity_score


    # --------------------------------------------------
    # SAVE YOLO RESULT IMAGE
    # --------------------------------------------------

    annotated = result.plot()

    result_filename = (
        "result_" + unique_name
    )

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        annotated
    )


    # --------------------------------------------------
    # SEND RESULTS TO HTML
    # --------------------------------------------------

    return render_template(

        "index.html",

        comparison=comparison,

        quantity_score=round(
            quantity_score,
            2
        ),

        position_score=(
            round(position_score, 2)
            if position_score is not None
            else None
        ),

        overall_score=round(
            overall_score,
            2
        ),

        uploaded_image=(
            "uploads/" + unique_name
        ),

        result_image=(
            "results/" + result_filename
        ),

        analysis_mode=analysis_mode
    )


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    # Render provides its own PORT automatically.
    # Locally/Colab it will use 10000 if PORT is absent.
    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
