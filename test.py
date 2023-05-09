import cv2
import numpy as np

# Load the background and the six frames
background = cv2.imread('background.jpg')

frames = []
for i in range(1, 2):
    frames.append(cv2.imread(f'f{i}.jpg'))

# Convert to grayscale
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
frames_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

# Subtract the background from each frame
output = np.zeros_like(background_gray)
for frame_gray in frames_gray:
    diff = cv2.absdiff(frame_gray, background_gray)
    _, mask = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
    output = cv2.add(output, mask)

# Show the output image
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
