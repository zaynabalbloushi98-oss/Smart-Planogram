from flask import Flask, render_template, request
import cv2
import numpy as np
import onnxruntime as ort
import os
import uuid
import hashlib

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.onnx")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# DEFAULT / FALLBACK PLANOGRAM
# ============================================================

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
# Must match the YOLO training class order.
CLASS_NAMES = [
    "fanta_orange",
    "pepsi_zero_sugar",
    "redbull"
]

DISPLAY_NAMES = {
    "pepsi_zero_sugar": "Pepsi Zero Sugar",
    "fanta_orange": "Fanta Orange",
    "redbull": "Red Bull"
}

CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

MIN_IMAGE_WIDTH = 500
MIN_IMAGE_HEIGHT = 350


# ============================================================
# LOAD ONNX MODEL
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded successfully", flush=True)
print("Input:", input_name, flush=True)


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
# PRODUCT DETECTION
# ============================================================

def detect_products(image):
    input_tensor, original_w, original_h = preprocess_image(image)

    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    predictions = outputs[0]
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

    indexes = nms(
        boxes,
        scores,
        IOU_THRESHOLD
    )

    detections = []

    for i in indexes:
        class_id = class_ids[i]

        if class_id >= len(CLASS_NAMES):
            continue

        x1, y1, x2, y2 = boxes[i]

        product = CLASS_NAMES[class_id]

        center_x = (x1 + x2) / 2

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
# BUILD PLANOGRAM FROM UPLOADED IMAGE
# ============================================================

def build_planogram_from_detections(detections):
    dynamic_expected = {}

    for product in CLASS_NAMES:
        product_detections = [
            detection
            for detection in detections
            if detection["product"] == product
        ]

        if not product_detections:
            continue

        quantity = len(product_detections)

        positions = [
            detection["zone"]
            for detection in product_detections
        ]

        expected_position = max(
            set(positions),
            key=positions.count
        )

        dynamic_expected[product] = {
            "quantity": quantity,
            "position": expected_position,
            "display": DISPLAY_NAMES[product]
        }

    return dynamic_expected
# ============================================================
# PLANOGRAM COMPARISON
# ============================================================

def analyze_planogram(detections, expected_config=None):
    if expected_config is None:
        expected_config = EXPECTED

    results = []

    correct_quantity_products = 0
    correct_position_products = 0

    messages = []

    for product, expected_data in expected_config.items():
        expected_qty = expected_data["quantity"]
        expected_position = expected_data["position"]
        display_name = expected_data["display"]

        product_detections = [
            detection
            for detection in detections
            if detection["product"] == product
        ]

        detected_qty = len(product_detections)

        detected_positions = [
            detection["zone"]
            for detection in product_detections
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
            "detected_positions": (
                ", ".join(detected_positions)
                if detected_positions
                else "-"
            ),
            "position_status": position_status,
            "missing_quantity": missing_qty
        })

    # ========================================================
    # COMPLIANCE
    # ========================================================

    total_products = len(expected_config)

    if total_products == 0:
        quantity_compliance = 0
        position_compliance = 0
        overall_compliance = 0

    else:
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

    # --------------------------------------------------------
    # CHECK BOTH UPLOADS
    # --------------------------------------------------------

    if "planogram" not in request.files or "image" not in request.files:
        return (
            "Please upload both a planogram image and a shelf image",
            400
        )

    planogram_file = request.files["planogram"]
    shelf_file = request.files["image"]

    if planogram_file.filename == "" or shelf_file.filename == "":
        return "Please select both images", 400

    # ========================================================
    # SAVE AND ANALYZE PLANOGRAM
    # ========================================================

    planogram_filename = (
        f"planogram_{uuid.uuid4().hex}.jpg"
    )

    planogram_path = os.path.join(
        UPLOAD_FOLDER,
        planogram_filename
    )

    planogram_file.save(planogram_path)

    planogram_image = cv2.imread(planogram_path)

    if planogram_image is None:
        return "Unable to read uploaded planogram image", 400

    planogram_height, planogram_width = (
        planogram_image.shape[:2]
    )

    if (
        planogram_width < MIN_IMAGE_WIDTH
        or planogram_height < MIN_IMAGE_HEIGHT
    ):
        return """
        <h2>Planogram image quality is too low</h2>
        <p>
            Please upload a higher-resolution planogram image
            for accurate product detection.
        </p>
        <p>
            Minimum recommended resolution: 500 × 350 pixels.
        </p>
        """, 400

    planogram_detections = detect_products(
        planogram_image
    )

    print(
        "=== PLANOGRAM DETECTIONS ===",
        flush=True
    )
    print(
        planogram_detections,
        flush=True
    )

    if not planogram_detections:
        return """
        <h2>No products detected in the planogram</h2>
        <p>
            Please upload a clear planogram image containing
            the supported products.
        </p>
        """, 400

    dynamic_expected = (
        build_planogram_from_detections(
            planogram_detections
        )
    )

    print(
        "=== DYNAMIC EXPECTED ===",
        flush=True
    )
    print(
        dynamic_expected,
        flush=True
    )

    # ========================================================
    # SAVE AND ANALYZE SHELF IMAGE
    # ========================================================

    shelf_filename = (
        f"{uuid.uuid4().hex}.jpg"
    )

    shelf_path = os.path.join(
        UPLOAD_FOLDER,
        shelf_filename
    )

    shelf_file.save(shelf_path)

    # --------------------------------------------------------
    # IMAGE HASH FOR DEBUGGING
    # --------------------------------------------------------

    with open(shelf_path, "rb") as f:
        image_hash = hashlib.sha256(
            f.read()
        ).hexdigest()

    print(
        "=== IMAGE SHA256 ===",
        flush=True
    )
    print(
        image_hash,
        flush=True
    )

    shelf_image = cv2.imread(
        shelf_path
    )

    if shelf_image is None:
        return "Unable to read uploaded shelf image", 400

    shelf_height, shelf_width = (
        shelf_image.shape[:2]
    )

    if (
        shelf_width < MIN_IMAGE_WIDTH
        or shelf_height < MIN_IMAGE_HEIGHT
    ):
        return """
        <h2>Image quality is too low</h2>
        <p>
            Please upload a higher-resolution shelf image
            for accurate product detection.
        </p>
        <p>
            Minimum recommended resolution: 500 × 350 pixels.
        </p>
        """, 400

    # --------------------------------------------------------
    # SHELF DETECTION
    # --------------------------------------------------------

    detections = detect_products(
        shelf_image
    )

    print(
        "=== SHELF DETECTIONS DEBUG ===",
        flush=True
    )
    print(
        detections,
        flush=True
    )

    # --------------------------------------------------------
    # DRAW SHELF DETECTIONS
    # --------------------------------------------------------

    result_image = draw_detections(
        shelf_image,
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

    # ========================================================
    # COMPARE SHELF AGAINST UPLOADED PLANOGRAM
    # ========================================================

    (
        results,
        messages,
        quantity_compliance,
        position_compliance,
        overall_compliance
    ) = analyze_planogram(
        detections,
        dynamic_expected
    )

    # ========================================================
    # RESULT PAGE
    # ========================================================

    return render_template(
        "result.html",

        original_image=f"uploads/{shelf_filename}",

        result_image=f"uploads/{result_filename}",

        results=results,

        messages=messages,

        quantity_compliance=round(
            quantity_compliance,
            2
        ),

        position_compliance=round(
            position_compliance,
            2
        ),

        overall_compliance=round(
            overall_compliance,
            2
        )
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
