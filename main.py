import cv2
from ultralytics import YOLO
import numpy as np
# Define the object name dictionary
object_labels = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "street sign", 12: "stop sign", 13: "parking meter",
    14: "bench", 15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep",
    20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    25: "hat", 26: "backpack", 27: "umbrella", 28: "shoe", 29: "eye glasses",
    30: "handbag", 31: "tie", 32: "suitcase", 33: "frisbee", 34: "skis",
    35: "snowboard", 36: "sports ball", 37: "kite", 38: "baseball bat",
    39: "baseball glove", 40: "skateboard", 41: "surfboard", 42: "tennis racket",
    43: "bottle", 44: "plate", 45: "wine glass", 46: "cup", 47: "fork",
    48: "knife", 49: "spoon", 50: "bowl", 51: "banana", 52: "apple",
    53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot", 57: "hot dog",
    58: "pizza", 59: "donut", 60: "cake", 61: "chair", 62: "couch",
    63: "potted plant", 64: "bed", 65: "mirror", 66: "dining table",
    67: "window", 68: "desk", 69: "toilet", 70: "door", 71: "tv",
    72: "laptop", 73: "mouse", 74: "remote", 75: "keyboard", 76: "cell phone",
    77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator",
    82: "blender", 83: "book", 84: "clock", 85: "vase", 86: "scissors",
    87: "teddy bear", 88: "hair drier", 89: "toothbrush", 90: "hair brush"
}

# cap = cv2.VideoCapture(1)  # For Webcam
cap = cv2.VideoCapture("video.mp4")  # For Video
model = YOLO("yolov8n.pt")  # Model Path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device="mps")  # Detects Objects in Frame(with gpu)
    # results = model(frame) #without gpu
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    confidences = np.array(result.boxes.conf.cpu(), dtype="float")

    for cls, bbox, confidence in zip(classes, bboxes, confidences):
        (x1, y1, x2, y2) = bbox
        object_name = object_labels.get(cls, "Unknown")  # Get object name from dictionary
        confidence_percentage = f"{confidence * 100:.2f}%"
        label = f"{object_name} {confidence_percentage}"
        
        # Calculate text size for creating the text container
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        
        # Create a rectangle for the text container above the bounding box
        text_bg_color = (0, 0, 0)
        text_fg_color = (200, 200, 0)
        padding = 15
        
        # Calculate the coordinates for the text container
        x1_text = x1
        y1_text = y1 - label_height - padding
        x2_text = x1 + label_width + 2 * padding
        y2_text = y1
        
        # Create the text container
        cv2.rectangle(frame, (x1_text, y1_text), (x2_text, y2_text), text_bg_color, -1)
        
        # Draw the bounding box and the text
        cv2.rectangle(frame, (x1, y1), (x2, y2), text_fg_color, 3)
        cv2.putText(frame, label, (x1 + padding, y1 - padding), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_fg_color, 2)


    cv2.imshow("Image", frame)  # Plays Frames One by One as Video
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to Quit
        break

cap.release()
cv2.destroyAllWindows()



'''
without gpu, speed ~ 59-65ms per frame..
with gpu, speed ~ 10-15ms per frame..

but with gpu, yolo isn't optimised(not for mac m1("mps device")).
so for me, without gpu is better.. also datasets are not that diverse..apart from person, vehicle class, everything is not that accurate...

'''
