import cv2
import subprocess
import os
import sys
import numpy as np

# Default image pulled in if no argument provided
defaultPath = 'TestImages/group.jpg'

# Storage for all detected faces in input image
face_rectangles = []

# selected[0]: x start, selected[1]: y start, selected[2]: x end, selected[3]: y end
selected = None

# Global storage for generated face
generatedFace = None # The actual image of the face itself, after masking the background
generatedFaceOriginal = None # A copy of the original generated face, to revert changed made in editing
generatedFull = None # The generated face but with padding added around it so it is the same size as the base image

# Load the cascade for face classification
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Padding for facial detection
padding = -.05

# Radius for drawing black circle around generated/over original face
backgroundRadius = -1

# User has clicked within image, check if it was within a detected face
def clickDetected(event, clicked_x, clicked_y, flags, param):
    global selected
    global backgroundRadius
    global generatedFull
    if event == cv2.EVENT_LBUTTONDOWN:
        for rect in face_rectangles: # Check all rectangles to see which was chosen
            # Turn this face rectangle green
            cv2.rectangle(scratchImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            # Use clicked within this detected face
            if rect[0] <= clicked_x <= rect[2] and rect[1] <= clicked_y <= rect[3]:
                selected = rect # Remember selected face
                backgroundRadius = int((selected[2] - selected[0]) / 2.2) # Set radius as a function of the selected size
                generatedFull = None # Set generated to none

# Use our trained stylegan2 model to generate a fake face
def GenerateFace():
    # Run command to generate face and get output so we know the image name
    output = subprocess.run('stylegan2_pytorch.exe --generate --name FFRSCCREP --num_image_tiles 1', stdout=subprocess.PIPE).stdout.decode('utf-8')
    start = output.find('at ./') + 5 # Get start of filepath
    end = output.find('\r', start) # Get end of filepath
    filepath = output[start:end]
    filepath += '-0-ema.jpg' # Rest of name to get the best result image
    Generated = cv2.imread(filepath) # Get generated face image
    # Remove all images from directory
    resultdir = 'results/FFRSCCREP'
    for f in os.listdir(resultdir):
        os.remove(os.path.join(resultdir, f))
    return Generated

# Perform background replacement so the image looks more natural
# Done by identifying the center of the face via face recognition and then adding a circular black mask around the face
def BackgroundReplacement(Generated):
    # Run facial recognition on generated face
    img = Generated.copy()
    gengray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gengray, 1.1, 4)
    roi = None
    for (x2, y2, w2, h2) in face:
        paddings = .001
        x2 -= int(w2 * paddings)
        w2 += int(w2 * paddings) * 2
        y2 -= int(h2 * paddings)
        h2 += int(h2 * paddings) * 2
        roi = img[y2 + 2:y2 + h2 - 1, x2 + 2:x2 + w2 - 1]
    # Resize the region of interest to be the same size as the selected face
    roi = cv2.resize(roi, (selected[2] - selected[0], selected[3] - selected[1]))
    # Create black circular mask around face
    height, width, depth = roi.shape
    nu_height = height/2
    nu_width = width/2
    nu_depth = depth/2
    int_height = int(nu_height)
    int_width = int(nu_width)
    int_depth = int(nu_depth)
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (int_width, int_height), backgroundRadius, 1, thickness=-1)
    masked_data = cv2.bitwise_and(roi, roi, mask=circle_img)
    return masked_data

# Add padding to generated face so that it is the same size as the base image
# This allows the bitwise XOR to paste the face onto the image
def FitFace(Generated):
    padding_left = selected[0]
    padding_top = selected[1]
    padding_right = originalImg.shape[1] - selected[2]
    padding_bottom = originalImg.shape[0] - selected[3]
    final = cv2.copyMakeBorder(Generated, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return final

# Generate a face, perform background replacement, fit it tot he image, place it in global storage so it can be accessed later
def ReplaceFace():
    global selected
    global generatedFace
    global generatedFaceOriginal
    global generatedFull
    Generated = GenerateFace()
    # There is a small chance to generate a face in which facial detection fails, while loop retries in these cases
    while True:
        try:
            Generated = BackgroundReplacement(Generated) # Attempt background replacement
            break
        except:
            Generated = GenerateFace()  # Generate another face and retry
    # Copy just the face with mask into appropriate variables
    generatedFace = Generated.copy()
    generatedFaceOriginal = Generated.copy()
    # Fit the generated face so it is the same size as the base image
    generatedFull = FitFace(Generated)

# Increase or decrease brightness of generated face
def ChangeGeneratedBrightness(change):
    for yg in range(generatedFace.shape[0]):
        for xg in range(generatedFace.shape[1]):
            for cg in range(generatedFace.shape[2]):
                if generatedFace[yg, xg, cg] != 0: # Don't increase value for black pixels to avoid brightening circle around face
                    generatedFace[yg, xg, cg] = np.clip(generatedFace[yg, xg, cg] + change, 0, 255)

# Increase or decrease contrast of generated face
def ChangeGeneratedContrast(change):
    for yg in range(generatedFace.shape[0]):
        for xg in range(generatedFace.shape[1]):
            for cg in range(generatedFace.shape[2]):
                generatedFace[yg, xg, cg] = np.clip(change * generatedFace[yg, xg, cg], 0, 255)

# Read the input image
filename = defaultPath
scratchImg = None
if len(sys.argv) > 1: # At least 1 argument provided
    filename = sys.argv[1] # Set filepath to argument
try:
    originalImg = cv2.imread(filename) # If it is a .png, will be read in with transparency
    scratchImg = originalImg.copy()  # Copy to not modify original image, used to put face detection and faces over
except:
    print("Problem reading in image from argument, please make sure it's a valid path to an image.")
    exit()
# Convert into grayscale
gray = cv2.cvtColor(scratchImg, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces: # x = start x position, y = start y position, w = width, h = height
    # We want to add some padding to these rectangles so that they will encompass the entire head, not just the face
    x -= int(w * padding)
    w += int(w * padding) * 2
    y -= int(h * padding)
    h += int(h * padding) * 2
    # Add to detected list
    face_rectangles.append((x, y, x+w, y+h))
    cv2.rectangle(scratchImg, (x, y), (x+w, y+h), (0, 255, 0), 2)
# Display the output
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', clickDetected, scratchImg)
cv2.imshow('Image', scratchImg)

while 1:
    # Make version of image with generated face placed over, if applicable
    # Don't want to put it on scratchImg yet if it has not been saved
    tempImg = scratchImg.copy()
    if generatedFull is not None:
        center = (selected[0] + int((selected[2] - selected[0]) / 2), selected[1] + int((selected[3] - selected[1]) / 2))
        cv2.circle(tempImg, center, backgroundRadius, (0, 0, 0), -1)
        tempImg = cv2.bitwise_xor(tempImg, generatedFull)
    cv2.imshow('Image', tempImg)
    k = cv2.waitKey(10) & 0xFF
    if selected is not None:
        cv2.rectangle(scratchImg, (selected[0], selected[1]), (selected[2], selected[3]), (255, 0, 0), 2)
        if k == 71 or k == 103: # G, generate face
            ReplaceFace()
        if generatedFull is not None:
            if k == 83 or k == 115: # S, save image with modified faces
                # Save version of image with face pasted over
                saveImg = originalImg.copy() # Copy original, unmodified image so rectangles will not be there
                # Add circles to images
                center = (selected[0] + int((selected[2] - selected[0]) / 2), selected[1] + int((selected[3] - selected[1]) / 2))
                cv2.circle(saveImg, center, backgroundRadius, (0, 0, 0), -1)
                cv2.circle(scratchImg, center, backgroundRadius, (0, 0, 0), -1)
                # Put face onto images
                saveImg = cv2.bitwise_xor(saveImg, generatedFull)
                scratchImg = cv2.bitwise_xor(scratchImg, generatedFull)
                originalImg = saveImg.copy() # Save changes to original image as well, so future saves will not overwrite
                # Save and reset appropriate variables
                cv2.imwrite(filename.replace(".jpg", "-modified.jpg").replace(".png", "-modified.png"), saveImg)
                selected = None
                generatedFull = None
                # Make all rectangles green again
                for rect in face_rectangles:
                    cv2.rectangle(scratchImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            elif k == 66 or k == 98: # B, increase brightness
                ChangeGeneratedBrightness(10)
                generatedFull = FitFace(generatedFace)
            elif k == 86 or k == 118: # V, decrease brightness
                ChangeGeneratedBrightness(-10)
                generatedFull = FitFace(generatedFace)
            elif k == 67 or k == 99: # C, increase contrast
                ChangeGeneratedContrast(1.1)
                generatedFull = FitFace(generatedFace)
            elif k == 88 or k == 120: # X, decrease contrast
                ChangeGeneratedContrast(0.9)
                generatedFull = FitFace(generatedFace)
            elif k == 89 or k == 121: # Y, horizontally flip image
                generatedFace = cv2.flip(generatedFace, 1)
                generatedFull = FitFace(generatedFace)
            elif k == 82 or k == 114: # R, return generated to original form
                generatedFace = generatedFaceOriginal.copy()
                generatedFull = FitFace(generatedFace)
    if k == 27: # Escape, close program
        break

cv2.destroyAllWindows()
