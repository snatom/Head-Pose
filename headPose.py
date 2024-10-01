#!/usr/bin/env python

import cv2
import numpy as np

# Welcome
print("Welcome to the Head Pose Estimation Script")

# Read Image
image_path = "headPose.jpg"
im = cv2.imread(image_path)

# Check if the image was successfully loaded
if im is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

size = im.shape

# 2D image points. If you change the image, you need to adjust the points accordingly
image_points = np.array([
    (359, 391),     # Nose tip
    (399, 561),     # Chin
    (337, 297),     # Left eye left corner
    (513, 301),     # Right eye right corner
    (345, 465),     # Left mouth corner
    (453, 469)      # Right mouth corner
], dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals
focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

print(f"Camera Matrix :\n {camera_matrix}")
print("Calculating head pose...")

# Assuming no lens distortion
dist_coeffs = np.zeros((4, 1))

# Solve PnP (Perspective-n-Point) to obtain rotation and translation vectors
(success, rotation_vector, translation_vector) = cv2.solvePnP(
    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

if not success:
    print("Error: SolvePnP failed.")
    exit()

print(f"Rotation Vector:\n {rotation_vector}")
print(f"Translation Vector:\n {translation_vector}")

# Project a 3D point (0, 0, 1000.0) onto the image plane to visualize direction
(nose_end_point2D, jacobian) = cv2.projectPoints(
    np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

# Draw the key points
for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

# Draw a line representing the nose direction
p1 = (int(image_points[0][0]), int(image_points[0][1]))  # Nose tip
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))  # Projected nose direction

cv2.line(im, p1, p2, (255, 0, 0), 2)

# Display the result
cv2.imshow("Output", im)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Head pose estimation completed successfully.")
