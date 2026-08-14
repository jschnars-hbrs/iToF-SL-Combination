import cv2 as cv
import os
SL_dir = "/mnt/ImportantCode/Masterprojekt_1/Pictures/Test/SL"
out_dir = os.path.join(os.path.dirname(SL_dir), "SL_inverted")

os.makedirs(out_dir,exist_ok=True)

for fname in os.listdir(SL_dir):
    if not fname.lower().endswith(".png"):
        continue
    img = cv.imread(os.path.join(SL_dir,fname),cv.IMREAD_UNCHANGED)
    cv.imwrite(os.path.join(out_dir,fname),cv.bitwise_not(img))

    