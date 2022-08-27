import sys
import numpy as np
import cv2 as cv

WINDOW_NAME = 'window'


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  


def draw_corner(filename):

    img = cv.imread(filename)
    #img.resize(640, 480)
    if len(img) == 0:
        print("Image Path Error!")
        return [],[]
    img = cv.resize(img, (480, 640) )
    points_add= []
    cv.namedWindow(WINDOW_NAME)
    
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 5, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    cv.destroyAllWindows()
    
    print('{} Points added'.format(len(points_add)))
    return img, np.array(points_add)

def DLT(points1, points2, normalized=False):

    A = []
    for i in range(4):
        u, v = points1[i,0], points1[i,1]
        u_, v_ = points2[i,0], points2[i,1]
        A.append([0, 0, 0, -u, -v, -1, v_*u, v_*v, v_])
        A.append([u, v, 1, 0, 0, 0, -u_*u, -u_*v, -u_])
    A = np.asarray(A)
    u, s, vh = np.linalg.svd(A)
    #A = U*S*V^H
    #print((vh.T[:,-1]/vh[-1,-1]).reshape(3,3))
    #print("opencv:\n {}".format(cv.findHomography(points1,points2)[0]))
    H = (vh.T[:,-1]/vh[-1,-1]).reshape(3,3)

    
    return H


def img_warping(img, H):
    ret_img = np.zeros(img.shape) # 640 480 3
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pos = np.dot(H, np.array([i,j,1]))
            pos = pos/pos[2]
            pos = np.round(pos)
            ret_img[i,j,:] = img[int(pos[1]),int(pos[0]),:]
    return ret_img.astype('uint8') #change to uint8 to show img properly

def show_result(img, filename):
    filetype = filename.split('.')[-1]
    filename = filename.replace('.'+filetype,'')
    print(filename)
    cv.imwrite("{}_result.jpg".format(filename),img)
    cv.imshow("result", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#  x---> 
# y 
# | 
# v
#
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[USAGE] python 2.py [IMAGE PATH]')
        sys.exit(1)

    check_corner_no = False
    while not check_corner_no:
        img, corners = draw_corner(sys.argv[1])
        if len(corners) == 4:
            check_corner_no = True
        else:
            print("Pleace click four corners clockwisely from the left top corner.")
    new_corners = np.array([[0,0], [0,479] , [639,479], [639,0]]) #four corner coordinate of new img
    
    H = DLT(corners, new_corners)
    inv_H = np.linalg.inv(H)
    warp_img = img_warping(img, inv_H)
    show_result(warp_img, sys.argv[1])
