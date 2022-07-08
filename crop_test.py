import cv2 as cv
import numpy as np

img_file = "C:/new_folder/votos.jpg"


frame = cv.imread(img_file)
copy = frame.copy()

copy_gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
thresh = 132
img_eroded = cv.erode(copy_gray,np.ones((3, 3), np.uint8),iterations=2)
# img_dilated = cv.dilate(thresh_img,(3,3),iterations=1)
ret,thresh_img = cv.threshold(img_eroded, thresh, 255, cv.THRESH_TRIANGLE)




cv.imwrite("C:/new_folder/resultado.jpg", thresh_img)
contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


i=0
new_img = frame.copy()
for c in contours:
    perimeter = cv.arcLength(c, True)
    
    approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        
        m = cv.moments(c)
        cx = int(m['m10']/m['m00'])
        cy = int(m['m01']/m['m00'])
        ratio_xy_centroid = cx/(cy+1)
        if ratio_xy_centroid>=0.10 and ratio_xy_centroid<=1.30 and perimeter>252:
            print(ratio_xy_centroid)
            print("ratio")
            # retorna uma tupla com x e y centrais, altura e largura, além do ângulo em relação ao centro a partir do contorno 
            rect = cv.minAreaRect(c)
            # retorna os vértices do retângulo 
            box = cv.boxPoints(rect)
            box = np.int64(box)

            cv.drawContours(new_img, [box], 0, (0, 0, 255), 1)
            cv.imwrite("C:/new_folder/contours.jpg", new_img)
            width = int(rect[1][0])
            height = int(rect[1][1])

            edge_pts_in_src_img = box.astype("float32")
 
            edge_pts_in_destine_img = np.array([[0, height],
                                [0, 0],
                                [width, 0],
                                [width, height]], dtype="float32")

            M = cv.getPerspectiveTransform(edge_pts_in_src_img, edge_pts_in_destine_img)

            warped = cv.warpPerspective(new_img, M, (width, height))
            warped = cv.rotate(warped,cv.ROTATE_90_CLOCKWISE)

            cv.imwrite("C:/new_folder/resultado"+str(i)+".jpg", warped)            
            i+=1
