import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

# Load image
img = cv2.imread('image1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display grayscale image
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.title("Grayscale Image")
plt.show()

# Apply bilateral filter to reduce noise
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

# Perform edge detection
edged = cv2.Canny(bfilter, 30, 200)

# Display edges
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title("Edge Detection")
plt.show()

# Find contours in the edged image
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)

# Sort the contours by area, and take the top 10
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

# Iterate through the contours to find a quadrilateral (4 points)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# Create a mask and draw the contour (the quadrilateral) on it
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Display the masked image with the detected contour
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("Masked Image")
plt.show()

# Get the coordinates of the region of interest (ROI) to crop
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

# Crop the region of interest (ROI) from the original image
cropped_image = gray[x1:x2+1, y1:y2+1]

# Display the cropped image
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped Image")
plt.show()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Perform OCR (optical character recognition) on the cropped image
result = reader.readtext(cropped_image)

# Display OCR results
print("OCR Result:", result)

# Extract the text from the result (first detected text block)
text = result[0][-2]

# Add the recognized text on the original image
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(location[0][0][0], location[0][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Draw a rectangle around the detected region (the contour)
res = cv2.rectangle(res, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

# Display the final image with detected text and bounding box
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title("Final Image with OCR and Bounding Box")
plt.show()
