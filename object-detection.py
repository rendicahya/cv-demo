import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)


def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


base_options = python.BaseOptions(model_asset_path="efficientdet.tflite")
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)
cap = cv2.VideoCapture()

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    cv2.imshow("Object Detection", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
