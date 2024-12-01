import cv2
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime

# Initialize counters
normal_box_count = 0
defective_box_count = 0
total_boxes = 0

# Keep track of already detected boxes to avoid duplicate counting
detected_boxes = []

#MQTT broker details
BROKER_ADDRESS = "pi5"
PORT = 1883
TOPIC = ""

# Webcam setup
cap = cv2.VideoCapture(0)  # Use webcam index 0
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def publish_integer(client, topic, integer):
    payload = str(integer)
    client.publish(topic, payload)
    print(f"Published well to {topic}")


def is_new_detection(new_box, detected_boxes, threshold=75):
    """
    Check if the new detection is significantly far from previously detected boxes.
    """
    x1, y1, w1, h1 = new_box
    for x2, y2, w2, h2 in detected_boxes:
        # Check overlap using Euclidean distance between centers
        if abs(x1 - x2) < threshold and abs(y1 - y2) < threshold:
            return False
    return True

def detect_sticker_color(frame, zone):
    """
    Detect red or green stickers within the specified zone.
    """
    global normal_box_count, defective_box_count, total_boxes, detected_boxes

    # Crop the frame to the zone of interest
    x, y, w, h = zone
    roi = frame[y:y + h, x:x + w]

    # Convert ROI to HSV for color filtering
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green in HSV
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])

    # Create masks for red and green
    red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
    red_mask = cv2.add(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)

    # Find contours for red and green regions
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process red contours
    for contour in red_contours:
        if cv2.contourArea(contour) > 500:  # Filter small areas
            x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
            new_box = (x_box, y_box, w_box, h_box)

            # Check if it's a new detection
            if is_new_detection(new_box, detected_boxes):
                normal_box_count += 1
                total_boxes += 1
                detected_boxes.append(new_box)  # Track this box
                save_image(roi[y_box:y_box + h_box, x_box:x_box + w_box], "normal")

            cv2.rectangle(roi, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)

    # Process green contours
    for contour in green_contours:
        if cv2.contourArea(contour) > 500:  # Filter small areas
            x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
            defective_box_count += 1
            save_image(roi[y_box:y_box + h_box, x_box:x_box + w_box], "defective")
            cv2.rectangle(roi, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 0, 255), 2)

    return roi

def save_image(roi, category):
    """
    Save the detected sticker region to a file with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{category}box{timestamp}.jpg"
    cv2.imwrite(filename, roi)

def display_counters(frame):
    """
    Overlay counters on the frame.
    """
    text = [
        f"Normal boxes: {normal_box_count}",
        f"Defective boxes: {defective_box_count}",
        f"Total boxes: {total_boxes}",
    ]
    for i, line in enumerate(text):
        cv2.putText(
            frame,
            line,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

client = mqtt.Client()
client.connect(BROKER_ADDRESS, PORT, 60)

print("Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Define zone of interest (square in the middle)
    height, width, _ = frame.shape
    zone_size = min(height, width) // 2  # Adjust size as needed
    x_center = width // 2
    y_center = height // 2
    zone = (x_center - zone_size // 2, y_center - zone_size // 2, zone_size, zone_size)

    # Draw the zone of interest on the frame (for visualization)
    x, y, w, h = zone
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect stickers within the zone of interest
    processed_zone = detect_sticker_color(frame, zone)

    # Display counters on the frame
    display_counters(frame)

    # Display the frame and the processed zone
    cv2.imshow("Box Detection", frame)
    cv2.imshow("Zone of Interest", processed_zone)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    TOPIC = "machine/good_boxes"
    publish_integer(client, TOPIC, normal_box_count)
    TOPIC = "machine/bad_boxes"
    publish_integer(client, TOPIC, defective_box_count)
    TOPIC = "machine/total_boxes"
    publish_integer(client, TOPIC, total_boxes)
    

# Release resources
cap.release()
cv2.destroyAllWindows()
client.disconect()

