import cv2
import numpy as np

#Image Processing

def normalize_brightness(img):
    img3=img.copy()
    img3 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    img3[:,:,2] = img3[:,:,2]*random_bright
    img3 = cv2.cvtColor(img3,cv2.COLOR_HSV2RGB)
    return img3

def h_equalize(img):
    img2=img.copy() 
    img2[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img2[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img2[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
    return img2

def resize_img(img):
    img2=img.copy()
    sc_y=0.4*np.random.rand()+1.0
    img2=cv2.resize(img, None, fx=1, fy=sc_y, interpolation = cv2.INTER_CUBIC)
    x,y, n = int(img2.shape[0]/2), int(img2.shape[1]/2), int(img_size/2)
    return img2

def crop(img):
    x,y, n = int(img.shape[0]/2), int(img.shape[1]/2), int(img_size/2-3)
    return img[(x-n):(x+n),(y-n):(y+n)]


def transform_image(img):
    # Rotation 
    global img_size
    img_size=img.shape[0]
    gaussian = cv2.GaussianBlur(img, (3,3), 20.0)
    img= cv2.addWeighted(img, 2, gaussian, -1, 0)

    angle_param=10
    ang = np.random.uniform(angle_param)-angle_param/2
    c_x,c_y,ch = img.shape    
    mat_rot = cv2.getRotationMatrix2D((c_y/2,c_x/2),ang,1.0)

    # Translation
    trans_param=5
    tr_x = trans_param*np.random.uniform()-trans_param/2
    tr_y = trans_param*np.random.uniform()-trans_param/2
    mat_trans = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    shear_param=3
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5+shear_param*np.random.uniform()-shear_param/2
    pt2 = 20+shear_param*np.random.uniform()-shear_param/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    mat_shear = cv2.getAffineTransform(pts1,pts2)
    
    
    #Warp Affine 
    img = cv2.warpAffine(img,mat_rot,(c_y,c_x))
    img = cv2.warpAffine(img,mat_trans,(c_y,c_x))
    img = cv2.warpAffine(img,mat_shear,(c_y,c_x))
    
    # Normalize Brightness  
    img2 = normalize_brightness(img)
    
    # Histogram Equalization
    img3=h_equalize(img2)
    
    return img3 #crop(resize_img(img3))#resize_img(crop(img3))