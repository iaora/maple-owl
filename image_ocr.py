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


#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#from PIL import Image

#url = 'https://cdn.discordapp.com/attachments/757732994374041702/797275536476864582/legends_cs.png'
#img_resp = get(url, stream=True).raw
#img = asarray(bytearray(img_resp.read()), dtype="uint8")
#img = cv2.imdecode(img, cv2.IMREAD_COLOR)

img = img=cv2.imread('/Users/jamie/discord_bot/overlap.png')
if img is None:
    print("Check file path")
img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
#cv2.imshow('OG img', img)
#cv2.waitKey(0)
"""
img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.GaussianBlur(img, (5, 5), 0)

(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite('bw_image.png', im_bw)
#thr = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
"""
# Load the aerial image and convert to HSV colourspace
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Define lower and uppper limits of what we call "brown"
lo=np.array([0,0,0])
hi=np.array([0,0,255])

# Mask image to only select browns
mask=cv2.inRange(hsv,lo,hi)

# Change image to red where we found brown
white_img = img.copy()
white_img[mask>0]=(255,255,255)
white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
white_img = cv2.GaussianBlur(white_img, (5, 5), 0)

(thresh, img_bw) = cv2.threshold(white_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imwrite('white_img.png', img_bw)
bw_image = img=cv2.imread('/Users/jamie/discord_bot/white_img.png')


img_info = pytesseract.image_to_string(img_bw, lang='eng')
print (img_info.splitlines())
#cv2.imshow('bw_image', bw_image)
#cv2.waitKey(0)
