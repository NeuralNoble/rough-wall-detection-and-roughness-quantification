import os
import cv2
import numpy as np

# Define paths
image_folder = 'fyp robot.v3i.yolov8/train/images'  # Folder containing your images
annotation_folder = 'fyp robot.v3i.yolov8/train/labels'  # Folder containing YOLO annotation txt files
smooth_folder = 'data_svr/smooth'  # Folder to save cropped smooth images
rough_folder = 'wall Inspection/data_svr/rough'  # Folder to save cropped rough images

# Create the output folders if they don't exist
os.makedirs(smooth_folder, exist_ok=True)
os.makedirs(rough_folder, exist_ok=True)


# Function to crop and save based on YOLO annotation
def crop_and_save(image_path, annotation_path):
    # Read the image
    image = cv2.imread(image_path)

    # Read the annotation file
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    # Process each object in the annotation file
    for line in lines:
        # Clean up the line, split by spaces and filter out any empty strings
        values = list(filter(None, line.strip().split()))

        # Ensure that the line has exactly 5 values
        if len(values) == 5:
            class_id, x_center, y_center, width, height = map(float, values)

            # Convert YOLO's normalized coordinates to pixel coordinates
            image_height, image_width, _ = image.shape
            x_center, y_center = int(x_center * image_width), int(y_center * image_height)
            width, height = int(width * image_width), int(height * image_height)

            # Calculate the top-left and bottom-right corner of the bounding box
            x_min = x_center - width // 2
            y_min = y_center - height // 2
            x_max = x_center + width // 2
            y_max = y_center + height // 2

            # Crop the image based on the bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Save the cropped image to the appropriate folder
            if class_id == 1:  # smooth class
                cv2.imwrite(
                    os.path.join(smooth_folder, f'{os.path.basename(image_path)}_smooth_{x_center}_{y_center}.jpg'),
                    cropped_image)
            elif class_id == 0:  # rough class
                cv2.imwrite(
                    os.path.join(rough_folder, f'{os.path.basename(image_path)}_rough_{x_center}_{y_center}.jpg'),
                    cropped_image)


# Iterate through all images and their corresponding annotation files
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    annotation_path = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

    # Process the image and its annotations
    if os.path.exists(annotation_path):
        crop_and_save(image_path, annotation_path)

print("Cropped images saved successfully.")