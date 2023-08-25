import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from matplotlib import pyplot as plt
from mediapipe.tasks.python import vision
import mediapipe as mp

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.labelbottom": False,
        "xtick.bottom": False,
        "ytick.labelleft": False,
        "ytick.left": False,
        "xtick.labeltop": False,
        "xtick.top": False,
        "ytick.labelright": False,
        "ytick.right": False,
    }
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
base_options = python.BaseOptions(model_asset_path="models/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    recognition_result = recognizer.recognize(image)
    annotated_image = image.numpy_view().copy()

    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0][0]
        title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        hand_landmarks = recognition_result.hand_landmarks

        for hand_landmarks in hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        cv2.putText(
            annotated_image,
            title,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_image,
            title,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imshow("Hand Landmarks Recognition", rgb_annotated_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
