from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Correct folder name
dataset_path = "datasets"

# Image names
image_files = [
    "001.jpg",
    "002.jpg",
    "003.jpg",
    "004.jpg",
    "005.jpg",
    "006.jpg",
    "007.jpg",
    "008.jpg"
]

# Process images
for filename in image_files:

    # Full image path
    image_path = os.path.join(dataset_path, filename)

    # Read image
    image = cv2.imread(image_path)

    # Check image
    if image is None:
        print(f"Image not found: {filename}")
        continue

    # Run detection
    results = model(image)

    # Draw results
    annotated_image = results[0].plot()

    # Print current image
    print(f"Analyzing: {filename}")

    # Show image
    cv2.imshow("SatVision AI", annotated_image)

    # Wait for key press
    key = cv2.waitKey(0)

    # Press q to quit
    if key == ord('q'):
        break

# Close windows
cv2.destroyAllWindows()