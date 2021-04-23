import cv2
import subprocess
import os
import sys
import numpy as np

# Storage for all detected faces in input image
face_rectangles = []

# selected[0]: x start, selected[1]: y start, selected[2]: x end, selected[3]: y end
selected = None
# Global storage for generated face
generated = None
generatedOriginal = None

# Load the cascade for face classification
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Padding for facial detection
padding = -.05

# Default image pulled in if no argument provided
defaultPath = 'TestImages/group.jpg'

# User has clicked within image, check if it was within a detected face
def clickDetected(event, clicked_x, clicked_y, flags, param):
    global selected
    global generated
    if event == cv2.EVENT_LBUTTONDOWN:
        for rect in face_rectangles: # Check all rectangles to see which was chosen
            # Turn this face rectangle green
            cv2.rectangle(scratchImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            # Use clicked within this detected face
            if rect[0] <= clicked_x <= rect[2] and rect[1] <= clicked_y <= rect[3]:
                selected = rect # Remember selected face
                generated = None # Set generated to none

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
# Current method to do this is simply to get just the center part of the face from generation and place that over just the center part of the original face
# (This is why our padding is negative, so that the generated area will be smaller)
def BackgroundReplacement(Generated):
    generatedGray = cv2.cvtColor(Generated, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(generatedGray, 1.1, 4) # Should only be one face detected
    (xg, yg, wg, hg) = (face[0][0], face[0][1], face[0][2], face[0][3]) # Get start x, y, width and height as in original face detection
    xg -= int(wg * padding)
    wg += int(wg * padding) * 2
    yg -= int(hg * padding)
    hg += int(hg * padding) * 2
    return Generated[yg:yg+hg, xg:xg+wg] # Cut out just the detected face

# Generate a face, perform background replacement, resize to selected face, place it in global storage so it can be accessed later
def ReplaceFace():
    global selected
    global generated
    global generatedOriginal
    Generated = GenerateFace()
    # There is a small chance to generate a face in which facial detection fails, while loop retries in these cases.
    while True:
        try:
            Generated = BackgroundReplacement(Generated)
            break
        except:
            Generated = GenerateFace() # Generate another face and retry
    Generated = cv2.resize(Generated, (selected[2]-selected[0], selected[3]-selected[1])) # Resize face to be same width and height as selected
    generated = Generated.copy()
    generatedOriginal = Generated.copy() # Save so can return to this original form if modifications are made

# Increase or decrease brightness of generated face
def ChangeGeneratedBrightness(change):
    for yg in range(generated.shape[0]):
        for xg in range(generated.shape[1]):
            for cg in range(generated.shape[2]):
                generated[yg, xg, cg] = np.clip(generated[yg, xg, cg] + change, 0, 255)

# Increase or decrease contrast of generated face
def ChangeGeneratedContrast(change):
    for yg in range(generated.shape[0]):
        for xg in range(generated.shape[1]):
            for cg in range(generated.shape[2]):
                generated[yg, xg, cg] = np.clip(change * generated[yg, xg, cg], 0, 255)

# Read the input image
filename = defaultPath
scratchImg = None
if len(sys.argv) > 1: # At least 1 argument provided
    filename = sys.argv[1] # Set filepath to argument
try:
    originalImg = cv2.imread(filename)
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
    if generated is not None:
        tempImg[selected[1]:selected[3], selected[0]:selected[2]] = generated
    cv2.imshow('Image', tempImg)
    k = cv2.waitKey(10) & 0xFF
    if selected is not None:
        cv2.rectangle(scratchImg, (selected[0], selected[1]), (selected[2], selected[3]), (255, 0, 0), 2)
        if k == 71 or k == 103: # G, generate face
            ReplaceFace()
        if generated is not None:
            if k == 83 or k == 115: # S, save image with modified faces
                # Save version of image with face pasted over
                saveImg = originalImg.copy() # Copy original, unmodified image so rectangles will not be there
                saveImg[selected[1]:selected[3], selected[0]:selected[2]] = generated # Put generated face there
                scratchImg[selected[1]:selected[3], selected[0]:selected[2]] = generated
                originalImg = saveImg.copy() # Save changes to original image as well, so future saves will not overwrite
                cv2.imwrite(filename.replace(".jpg", "-modified.jpg").replace(".png", "-modified.png"), saveImg)
                selected = None
                generated = None
                # Make all rectangles green again
                for rect in face_rectangles:
                    cv2.rectangle(scratchImg, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            elif k == 66 or k == 98: # B, increase brightness
                ChangeGeneratedBrightness(10)
            elif k == 86 or k == 118: # V, decrease brightness
                ChangeGeneratedBrightness(-10)
            elif k == 67 or k == 99: # C, increase contrast
                ChangeGeneratedContrast(1.1)
            elif k == 88 or k == 120: # X, decrease contrast
                ChangeGeneratedContrast(0.9)
            elif k == 82 or k == 114: # R, return generated to original form
                generated = generatedOriginal.copy()
    if k == 27: # Escape, close program
        break

cv2.destroyAllWindows()
