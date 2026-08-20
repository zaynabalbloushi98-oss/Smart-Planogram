from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from collections import Counter, defaultdict

import onnxruntime as ort
import numpy as np
import cv2
import os
import uuid


app = Flask(__name__)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.onnx"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# =========================================================
# ONNX MODEL
# =========================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# IMPORTANT:
# This order must match the YOLO training classes.
CLASS_NAMES = [
    "fanta_orange",
    "pepsi_zero_sugar",
    "redbull"
]


# =========================================================
# EXPECTED VISUAL PLANOGRAM
# =========================================================

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


# =========================================================
# POSITION FUNCTION
# =========================================================

def get_zone(center_x, image_width):

    ratio = center_x / image_width

    if ratio < (1 / 3):
        return "left"

    elif ratio < (2 / 3):
        return "center"

    else:
        return "right"


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    # Read original image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read uploaded image.")

    original_height, original_width = image.shape[:2]

    # Resize to YOLO input size
    resized = cv2.resize(
        image,
        (640, 640)
    )

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float and normalize 0–1
    input_tensor = rgb.astype(
        np.float32
    ) / 255.0

    # HWC → CHW
    input_tensor = np.transpose(
        input_tensor,
        (2, 0, 1)
    )

    # Add batch dimension
    input_tensor = np.expand_dims(
        input_tensor,
        axis=0
    )

    return (
        image,
        input_tensor,
        original_width,
        original_height
    )


# =========================================================
# ONNX YOLO DETECTION
# =========================================================

def run_detection(
    image_path,
    conf_threshold=0.50,
    iou_threshold=0.50
):

    (
        original_image,
        input_tensor,
        original_width,
        original_height
    ) = preprocess_image(
        image_path
    )


    # Run ONNX inference
    output = session.run(
        None,
        {
            input_name:
                input_tensor
        }
    )[0]


    # YOLOv8 typically returns:
    # (1, 7, 8400) for 3 classes
    output = np.squeeze(output)


    # Convert:
    # (7, 8400) → (8400, 7)
    if output.shape[0] < output.shape[1]:

        output = output.T


    boxes = []
    scores = []
    class_ids = []


    # Scaling from 640 back to original image
    x_scale = original_width / 640.0
    y_scale = original_height / 640.0


    for prediction in output:

        # First 4 values:
        # center_x, center_y, width, height
        x_center = prediction[0]
        y_center = prediction[1]
        width = prediction[2]
        height = prediction[3]


        # Remaining values are class scores
        class_scores = prediction[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )


        # Ignore low confidence detections
        if confidence < conf_threshold:
            continue


        # Convert xywh → xyxy
        x1 = (
            x_center - width / 2
        ) * x_scale

        y1 = (
            y_center - height / 2
        ) * y_scale

        x2 = (
            x_center + width / 2
        ) * x_scale

        y2 = (
            y_center + height / 2
        ) * y_scale


        # OpenCV NMS requires x, y, w, h
        box_width = x2 - x1
        box_height = y2 - y1


        boxes.append([
            int(x1),
            int(y1),
            int(box_width),
            int(box_height)
        ])

        scores.append(
            confidence
        )

        class_ids.append(
            class_id
        )


    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        conf_threshold,
        iou_threshold
    )


    detections = []


    if len(indices) > 0:

        indices = np.array(
            indices
        ).flatten()


        for index in indices:

            x, y, w, h = boxes[index]

            class_id = class_ids[index]

            confidence = scores[index]


            if class_id >= len(CLASS_NAMES):
                continue


            class_name = (
                CLASS_NAMES[class_id]
            )


            detections.append({

                "product":
                    class_name,

                "confidence":
                    confidence,

                "x1":
                    x,

                "y1":
                    y,

                "x2":
                    x + w,

                "y2":
                    y + h
            })


    return (
        original_image,
        detections
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# ANALYZE
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    if "image" not in request.files:

        return (
            "No image uploaded",
            400
        )


    file = request.files["image"]


    if file.filename == "":

        return (
            "No image selected",
            400
        )


    analysis_mode = request.form.get(
        "analysis_mode",
        "full"
    )


    # Safe unique filename
    safe_name = secure_filename(
        file.filename
    )

    unique_name = (
        f"{uuid.uuid4().hex}_"
        f"{safe_name}"
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )


    file.save(
        upload_path
    )


    # =====================================================
    # DETECTION
    # =====================================================

    image, detections = run_detection(
        upload_path,
        conf_threshold=0.50,
        iou_threshold=0.50
    )


    image_height, image_width = (
        image.shape[:2]
    )


    counts = Counter()

    positions = defaultdict(list)


    # =====================================================
    # PROCESS DETECTIONS
    # =====================================================

    for detection in detections:

        product = detection[
            "product"
        ]

        confidence = detection[
            "confidence"
        ]

        x1 = detection["x1"]
        y1 = detection["y1"]
        x2 = detection["x2"]
        y2 = detection["y2"]


        center_x = (
            x1 + x2
        ) / 2


        zone = get_zone(
            center_x,
            image_width
        )


        counts[
            product
        ] += 1


        positions[
            product
        ].append(
            zone
        )


        # =================================================
        # DRAW BOX
        # =================================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        label = (
            f"{product} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            image,
            label,
            (
                x1,
                max(
                    y1 - 8,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    # =====================================================
    # PLANOGRAM COMPARISON
    # =====================================================

    comparison = []


    total_expected = sum(
        item["expected_quantity"]
        for item
        in PLANOGRAM.values()
    )


    total_quantity_error = 0

    correct_position_total = 0


    for product, info in PLANOGRAM.items():

        expected_qty = (
            info["expected_quantity"]
        )


        expected_zone = (
            info["expected_position"]
        )


        detected_qty = (
            counts.get(
                product,
                0
            )
        )


        zones = positions.get(
            product,
            []
        )


        # ---------------------------------
        # QUANTITY STATUS
        # ---------------------------------

        if detected_qty == expected_qty:

            quantity_status = (
                "Correct"
            )

        elif detected_qty < expected_qty:

            quantity_status = (
                f"Missing "
                f"{expected_qty - detected_qty}"
            )

        else:

            quantity_status = (
                f"Extra "
                f"{detected_qty - expected_qty}"
            )


        total_quantity_error += abs(
            expected_qty -
            detected_qty
        )


        # ---------------------------------
        # POSITION
        # ---------------------------------

        if analysis_mode == "full":

            correct_in_zone = min(
                zones.count(
                    expected_zone
                ),
                expected_qty
            )


            correct_position_total += (
                correct_in_zone
            )


            if detected_qty == 0:

                position_status = (
                    "Product Missing"
                )

            else:

                misplaced = sum(

                    1

                    for zone in zones

                    if zone
                    != expected_zone
                )


                if misplaced == 0:

                    position_status = (
                        "Correct"
                    )

                else:

                    position_status = (
                        f"Misplaced "
                        f"{misplaced}"
                    )


            detected_position_text = (
                ", ".join(zones)
                if zones
                else "-"
            )


        else:

            position_status = (
                "Not Evaluated"
            )

            detected_position_text = (
                "N/A"
            )


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


    # =====================================================
    # COMPLIANCE
    # =====================================================

    quantity_score = max(
        0,
        (
            1 -
            total_quantity_error /
            total_expected
        ) * 100
    )


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

        overall_score = (
            quantity_score
        )


    # =====================================================
    # SAVE RESULT IMAGE
    # =====================================================

    result_filename = (
        "result_" +
        unique_name
    )


    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )


    cv2.imwrite(
        result_path,
        image
    )


    # =====================================================
    # RETURN PAGE
    # =====================================================

    return render_template(

        "index.html",

        comparison=comparison,

        quantity_score=round(
            quantity_score,
            2
        ),

        position_score=(
            round(
                position_score,
                2
            )
            if position_score
            is not None
            else None
        ),

        overall_score=round(
            overall_score,
            2
        ),

        uploaded_image=(
            "uploads/" +
            unique_name
        ),

        result_image=(
            "results/" +
            result_filename
        ),

        analysis_mode=analysis_mode
    )


# =========================================================
# START APP
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
