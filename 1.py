import sys
import numpy as np
import cv2 as cv

#command example:
#python .\1.py .\images\1-0.png .\images\1-2.png .\groundtruth_correspondences\correspondence_02.npy .\screenshot\2\ norm



def get_sift_correspondences(img1, img2, num, screenshot_path):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
        num: the number of sample points
        screenshot_path: the dir to save screenshot img

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)   #descriptors
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
       
        if m.distance < 0.75 * n.distance:
        
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:num]
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imwrite('{}/sample_{}_screen.jpg'.format(screenshot_path,num), img_draw_match)
    #cv.imshow('match', img_draw_match)
    #cv.waitKey(0)
    return points1, points2



def DLT(points1, points2):
    
    """
    Input: 
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    Output:
        H: 3x3 Homography matrix that tranform points1 to points2
    """
    A = []
    for i in range(len(points1)):
        u, v = points1[i,0], points1[i,1]
        u_, v_ = points2[i,0], points2[i,1]
        A.append([0, 0, 0, -u, -v, -1, v_*u, v_*v, v_])
        A.append([u, v, 1, 0, 0, 0, -u_*u, -u_*v, -u_])
    A = np.asarray(A)
    u, s, vh = np.linalg.svd(A)
    #A = U*S*V^H
    #print("===========================================")
    #print("myself:\n",(vh.T[:,-1]/vh[-1,-1]).reshape(3,3))
    #print("opencv:\n {}".format(cv.findHomography(selected_p1,selected_p2)[0]))
    #print("===========================================")
    H = (vh.T[:,-1]/vh[-1,-1]).reshape(3,3)

    
    return H

def L2_distance(a, b):
    return np.sqrt(np.sum((a-b) ** 2))

def points_normalization(points):
    """
    Input:
        points: the point set Nx2
    Output:
        T_inv: the tranform matrix from original points to normalized points
        ret_points: the normalized points
    """
    m, s = np.mean(points,0), np.std(points)
    s = s/np.sqrt(2)

    T = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    
        
    T_inv = np.linalg.inv(T)
    ret_points = np.matmul( T_inv, np.concatenate( (points.T,np.ones((1,points.shape[0]))) ) )
    ret_points = ret_points[0:2,:].T
    L2_norm = 0
    for n in ret_points:
        L2_norm += L2_distance(np.array([n[0],n[1]]),np.array([0,0]))
    print("average_distance to origin: ",L2_norm/len(ret_points))
    return T_inv, ret_points


#p_t = H*p_s 
def cal_error(H, p_s, p_t):
    err = 0
    for idx,p in enumerate(p_s):
        Hp_s = np.matmul(H, np.array([p[0],p[1],1]))
        #print("Hp_s:",Hp_s)
        err += np.sqrt((p_t[idx][0] - Hp_s[0]) ** 2 + (p_t[idx][1] - Hp_s[1]) ** 2)
    err = err / len(p_s)
    return err

if __name__ == '__main__':
    
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    screenshot_path = sys.argv[4]
    try:
        normalize = (sys.argv[5] == 'norm')
    except:
        normalize = False
    sample_no = [4, 8, 20, 50]
    for i in sample_no:
        print("sample {} points".format(i))
        points1, points2 = get_sift_correspondences(img1, img2, i, screenshot_path)
        if normalize:
            #print("points1 in normalization.")
            T1, points1 = points_normalization(points1)
            #print("points2 in normalization.")
            T2, points2 = points_normalization(points2)
            H = DLT(points1,points2)
            H = np.dot(np.linalg.inv(T2),np.dot(H,T1))
        else:
            H = DLT(points1,points2)
    
        print("Error between groudtruth: ",cal_error(H, gt_correspondences[0], gt_correspondences[1]))