import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic,mark_boundaries
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from skimage.io import imsave
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb


#read grayscale image
gray_im = cv2.imread('grayscale_im.jpg')

#split image into 100 segments
segments = slic(gray_im,n_segments=100,sigma=5,start_label=1)

sift = cv2.SIFT_create()

#initialize parameters   
ksize = 10
sigma = 5
theta = 1*np.pi/4
lamda = 1*np.pi/4
phi = 0
gamma = 1

kernel = cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)

#-------------------------------------Extract SIFT & GABOR features from each SUPERPIXEL-----------------------#

for segmentValue in np.unique(segments):    
    mask = np.zeros(gray_im.shape[:2],dtype='uint8')
    mask[segments == segmentValue] = 255
    #---------------------------------------------SIFT---------------------------------------------------------#
    features,descriptions = sift.detectAndCompute(cv2.bitwise_and(gray_im, gray_im, mask = mask),None)
    sift_im = cv2.drawKeypoints(cv2.bitwise_and(gray_im, gray_im, mask = mask),features,None)
    #---------------------------------------------GABOR--------------------------------------------------------#
    filt_img = cv2.filter2D(cv2.bitwise_and(gray_im, gray_im, mask = mask),cv2.CV_8UC3,kernel)
   

#------------------------------------------Colorize Grayscale Image--------------------------------------------#

#load image colorization model
model_path = 'ColorizationModel.model'
Image_Colorization_Model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=None,
                        compile=True)

arr_img=[]
input_img=img_to_array(load_img('arnold.jpg'))
input_img = resize(input_img ,(256,256))

#convert image to LAB color space
arr_img.append(input_img)
arr_img= np.array(arr_img, dtype=float)
arr_img = rgb2lab(1.0/255*arr_img)[:,:,:,0]
arr_img = arr_img.reshape(arr_img.shape+(1,))

colored_im = Image_Colorization_Model.predict(arr_img)

#convert image to RGB color space
colored_im = colored_im*128
img_result = np.zeros((256, 256, 3))
img_result[:,:,0] = arr_img[0][:,:,0]
img_result[:,:,1:] = colored_im[0]
imsave("colored_Image.png", lab2rgb(img_result))

plt.figure(1)
plt.title('Original Grayscale Image')
plt.imshow(gray_im)

#SIFT on original grayscale image
keypoints,descriptions = sift.detectAndCompute(gray_im,None)
sift_img = cv2.drawKeypoints(gray_im,keypoints,None)
plt.figure(2)
plt.title('SIFT Result')
plt.imshow(mark_boundaries(sift_img,segments))

#GABOR on original grayscale image
gabor_img = cv2.filter2D(gray_im,cv2.CV_8UC3,kernel)
plt.figure(3)
plt.title('GABOR Result')
plt.imshow(mark_boundaries(gabor_img,segments))

#print colorized image
plt.figure(4)
plt.title('Colorized Imaged')
plt.imshow(lab2rgb(img_result))
plt.show()
