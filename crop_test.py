import cv2 as cv
from cv2 import threshold
import numpy as np

img_file = "C:/users/prfev/Desktop/new_crop_test/votos.jpg"
#erode, dilate, binarization, find contours, crop

areas_list = []
four_points_list = []
contours_index = []


frame = cv.imread(img_file)
copy = frame.copy()

copy_gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
thresh = 140
ret,thresh_img = cv.threshold(copy_gray, thresh, 255, cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
i=0
new_img = frame.copy()
for c in contours:
    perimeter = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        area = cv.contourArea(c)
        m = cv.moments(c)
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
        ratio_xy_centroid = cx/cy
        if ratio_xy_centroid>=0.77 and ratio_xy_centroid<=0.95:
            four_points_list.append(approx)
            contours_index.append(i)

            approx = np.ndarray.tolist(approx)
            x1,y1 = approx[0][0]
            x2,y2 = approx[1][0]
            x3,y3 = approx[2][0]
            x4,y4 = approx[3][0]
            top_left_x = min([x1,x2,x3,x4])
            top_left_y = min([y1,y2,y3,y4])
            bot_right_x = max([x1,x2,x3,x4])
            bot_right_y = max([y1,y2,y3,y4])
            new_img = frame[top_left_y:bot_right_y, top_left_x:bot_right_x]
            cv.imwrite("C:/users/prfev/Desktop/new_crop_test/resultado"+str(i)+".jpg", new_img)            
            i+=1

# for i in contours_index:
#     mask = np.zeros_like(new_img) # Create mask where white is what we want, black otherwise
#     cv.drawContours(mask, contours, i, 255, -1) # Draw filled contour in mask
#     out = np.zeros_like(new_img) # Extract out the object and place into output image
#     out[mask == 255] = new_img[mask == 255]
#     (y, x) = np.where(mask == 255)
#     (topy, topx) = (np.min(y), np.min(x))
#     (bottomy, bottomx) = (np.max(y), np.max(x))
#     out = out[topy:bottomy+1, topx:bottomx+1]
#     cv.imwrite("C:/users/prfev/Desktop/new_crop_test/resultado"+str(i)+".jpg", out)


copy = cv.drawContours(copy, four_points_list, -1, (0,0,255),1)


cv.imwrite("C:/users/prfev/Desktop/new_crop_test/resultado.jpg", copy)