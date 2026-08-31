
from flask import Flask, render_template, request
import cv2
import numpy as np
import onnxruntime as ort
import os
import uuid

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# SMART PLANOGRAM SETTINGS
# ============================================================

# Expected arrangement:
# LEFT   = Pepsi Zero Sugar
# CENTER = Fanta Orange
# RIGHT  = Red Bull

EXPECTED = {
    "pepsi_zero_sugar": {
        "quantity": 4,
        "position": "left",
        "display": "Pepsi Zero Sugar"
    },
    "fanta_orange": {
        "quantity": 4,
        "position": "center",
        "display": "Fanta Orange"
    },
    "redbull": {
        "quantity": 4,
        "position": "right",
        "display": "Red Bull"
    }
}

# IMPORTANT:
# These must match the class order used when training the YOLO model.
CLASS_NAMES = [
    "fanta_orange",
    "pepsi_zero_sugar",
    "redbull"
]

CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

# ============================================================
# LOAD ONNX MODEL
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded successfully")
print("Input:", input_name)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image, size=640):

    original_h, original_w = image.shape[:2]

    resized = cv2.resize(image, (size, size))

    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))

    img = np.expand_dims(img, axis=0)

    return img, original_w, original_h


# ============================================================
# NMS
# ============================================================

def nms(boxes, scores, threshold=0.45):

    if len(boxes) == 0:
        return []

    boxes_xywh = []

    for box in boxes:

        x1, y1, x2, y2 = box

        boxes_xywh.append([
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1)
        ])

    indexes = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores,
        CONF_THRESHOLD,
        threshold
    )

    if len(indexes) == 0:
        return []

    return np.array(indexes).flatten().tolist()


# ============================================================
# DETECTION
# ============================================================

def detect_products(image):

    input_tensor, original_w, original_h = preprocess_image(image)

    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    predictions = outputs[0]

    # YOLOv8 ONNX normally returns:
    # (1, 4 + number_of_classes, 8400)

    predictions = np.squeeze(predictions)

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []

    scale_x = original_w / 640
    scale_y = original_h / 640

    for prediction in predictions:

        x, y, w, h = prediction[:4]

        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        x1 = (x - w / 2) * scale_x
        y1 = (y - h / 2) * scale_y

        x2 = (x + w / 2) * scale_x
        y2 = (y + h / 2) * scale_y

        boxes.append([
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    indexes = nms(boxes, scores, IOU_THRESHOLD)

    detections = []

    for i in indexes:

        class_id = class_ids[i]

        if class_id >= len(CLASS_NAMES):
            continue

        x1, y1, x2, y2 = boxes[i]

        product = CLASS_NAMES[class_id]

        center_x = (x1 + x2) / 2

        # Determine horizontal shelf zone
        if center_x < original_w / 3:
            zone = "left"

        elif center_x < (original_w * 2 / 3):
            zone = "center"

        else:
            zone = "right"

        detections.append({
            "product": product,
            "confidence": scores[i],
            "box": [x1, y1, x2, y2],
            "zone": zone
        })

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(image, detections):

    result = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        product = detection["product"]
        confidence = detection["confidence"]

        label = f"{product} {confidence:.2f}"

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return result


# ============================================================
# PLANOGRAM COMPARISON
# ============================================================

def analyze_planogram(detections):

    results = []

    correct_quantity_products = 0
    correct_position_products = 0

    messages = []

    for product, expected_data in EXPECTED.items():

        expected_qty = expected_data["quantity"]
        expected_position = expected_data["position"]
        display_name = expected_data["display"]

        product_detections = [
            d for d in detections
            if d["product"] == product
        ]

        detected_qty = len(product_detections)

        detected_positions = [
            d["zone"] for d in product_detections
        ]

        # ----------------------------------------------------
        # QUANTITY
        # ----------------------------------------------------

        if detected_qty == expected_qty:

            quantity_status = "Correct"
            missing_qty = 0
            correct_quantity_products += 1

        elif detected_qty < expected_qty:

            missing_qty = expected_qty - detected_qty

            quantity_status = f"Missing {missing_qty}"

            messages.append(
                f"Quantity is missing {missing_qty} {display_name}. "
                f"Please provide {missing_qty} {display_name}. "
                f"The required quantity on the shelf is {expected_qty}."
            )

        else:

            extra_qty = detected_qty - expected_qty
            missing_qty = 0

            quantity_status = f"Extra {extra_qty}"

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        if detected_qty == 0:

            position_status = "Product Missing"

        else:

            misplaced = sum(
                1
                for position in detected_positions
                if position != expected_position
            )

            if misplaced == 0:

                position_status = "Correct"
                correct_position_products += 1

            else:

                position_status = f"Misplaced {misplaced}"

        results.append({

            "product": display_name,

            "expected_quantity": expected_qty,

            "detected_quantity": detected_qty,

            "quantity_status": quantity_status,

            "expected_position": expected_position.capitalize(),

            "detected_positions":
                ", ".join(detected_positions)
                if detected_positions else "-",

            "position_status": position_status,

            "missing_quantity": missing_qty
        })

    # ========================================================
    # COMPLIANCE
    # ========================================================

    total_products = len(EXPECTED)

    quantity_compliance = (
        correct_quantity_products / total_products
    ) * 100

    position_compliance = (
        correct_position_products / total_products
    ) * 100

    overall_compliance = (
        quantity_compliance + position_compliance
    ) / 2

    return (
        results,
        messages,
        quantity_compliance,
        position_compliance,
        overall_compliance
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# ANALYZE
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:

        return "No image uploaded", 400

    file = request.files["image"]

    if file.filename == "":

        return "No image selected", 400

    # Unique filename prevents old images from being cached
    filename = f"{uuid.uuid4().hex}.jpg"

    original_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(original_path)

    image = cv2.imread(original_path)

    if image is None:

        return "Unable to read uploaded image", 400

    # Detection
    detections = detect_products(image)
    print("=== DETECTIONS DEBUG ===", flush=True)
print(detections, flush=True)

    # Draw bounding boxes
    result_image = draw_detections(
        image,
        detections
    )

    result_filename = (
        f"result_{uuid.uuid4().hex}.jpg"
    )

    result_path = os.path.join(
        UPLOAD_FOLDER,
        result_filename
    )

    cv2.imwrite(
        result_path,
        result_image,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    # Planogram comparison
    (
        results,
        messages,
        quantity_compliance,
        position_compliance,
        overall_compliance
    ) = analyze_planogram(detections)

    return render_template(
        "result.html",

        original_image=
            f"uploads/{filename}",

        result_image=
            f"uploads/{result_filename}",

        results=results,

        messages=messages,

        quantity_compliance=
            round(quantity_compliance, 2),

        position_compliance=
            round(position_compliance, 2),

        overall_compliance=
            round(overall_compliance, 2)
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
