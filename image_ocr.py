import pytesseract
import urllib
import cv2
from io import BytesIO
from requests import get
import numpy as np

# Define some colours for readability - these are in OpenCV **BGR** order - reverse them for PIL
red   = [0,0,255]
green = [0,255,0]
blue  = [255,0,0]
white = [255,255,255]
black = [0,0,0]


img = img=cv2.imread('/Users/jamie/discord_bot/overlap.png')
if img is None:
    print("Check file path")
img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
# Load the image and convert to HSV colourspace
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Define lower and uppper limits of what we call "black"
lo=np.array([0,0,0])
hi=np.array([0,0,255])

# Mask image to only select black
mask=cv2.inRange(hsv,lo,hi)

# Change image to white where we found black
white_img = img.copy()
white_img[mask>0]=(255,255,255)

# Convert image to greyscale binary 
white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
white_img = cv2.GaussianBlur(white_img, (5, 5), 0)
(thresh, img_bw) = cv2.threshold(white_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Write image to disk 
cv2.imwrite('white_img.png', img_bw)
bw_image = img=cv2.imread('/Users/jamie/discord_bot/white_img.png')

# Process image to text
img_info = pytesseract.image_to_string(img_bw, lang='eng')
print (img_info.splitlines())
#cv2.imshow('bw_image', bw_image)
#cv2.waitKey(0)
