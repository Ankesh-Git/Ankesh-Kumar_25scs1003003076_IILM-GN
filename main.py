import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# ------------------------------------------
# STEP 1 : Start Program
# ------------------------------------------
print("MICROPLASTIC DETECTION SYSTEM")
print("This program detects microplastic particles in a microscope water sample image.\n")

# ------------------------------------------
# STEP 2 : Input
# ------------------------------------------
image_path = input("Enter the path of the microscope image: ")

if not os.path.exists(image_path):
    print("Error: File not found. Please provide a valid image path.")
    exit()

# ------------------------------------------
# STEP 3 : Image Addition
# ------------------------------------------
print("\nImage uploaded successfully. Processing...\n")

# ------------------------------------------
# STEP 4 : Image Preprocessing
# ------------------------------------------
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Binary threshold
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

# ------------------------------------------
# STEP 5 : Particle Observation
# ------------------------------------------
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

particle_count = 0
output_image = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Microplastic area range (20–2000 pixels)
    if 20 <= area <= 2000:
        particle_count += 1
        cv2.drawContours(output_image, [cnt], -1, (0, 0, 255), 2)  # Red boundaries

print(f"Total Microplastic-like particles detected: {particle_count}")

# ------------------------------------------
# STEP 6 : Pollution Level Analysis
# ------------------------------------------
if particle_count < 30:
    status = "Safe"
    color = "Green"
elif 30 <= particle_count < 80:
    status = "Moderate Pollution"
    color = "Orange"
else:
    status = "Critical Pollution"
    color = "Red"

print(f"Pollution Status: {status} (Color Code: {color})")

# ------------------------------------------
# STEP 7 : Display Results
# ------------------------------------------
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Particles: {particle_count} | Status: {status}")
plt.axis("off")
plt.show()

# ------------------------------------------
# STEP 8 : Data Record (Logging)
# ------------------------------------------
log_file = "microplastic_log.txt"
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(log_file, "a") as f:
    f.write(f"\n[{timestamp}] File: {os.path.basename(image_path)} | Particles: {particle_count} | Status: {status}")

print(f"\n✔ Log saved to {log_file}")

# ------------------------------------------
# STEP 9 : End
# ------------------------------------------
print("\nProgram completed successfully.")
