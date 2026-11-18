from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
import numpy as np
from keras.layers import Conv2D,UpSampling2D
from keras.models import Sequential
from skimage.segmentation import slic
import cv2

#read dataset
dataset = 'Path to dataset'
#https://www.kaggle.com/prasunroy/natural-images ----> person folder/directory

#rescale images
resc_dataset = ImageDataGenerator(rescale=1./255)
training_set = resc_dataset.flow_from_directory(dataset,
                                                target_size=(256,256),
                                                batch_size=340,
                                                class_mode=None)

#initialize parameters   
ksize = 10
sigma = 5
theta = 1*np.pi/4
lamda = 1*np.pi/4
phi = 0
gamma = 1

kernel = cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv2.CV_32F)

sift = cv2.SIFT_create()

train_x = []
train_y = []
SIFT_Features = []
GABOR_Features = []
for i in training_set[0]:

    #convert image to LAB space  
    LAB_img = rgb2lab(i)
        
    train_x.append(LAB_img[:,:,0]) #Get L values
    train_y.append(LAB_img[:,:,1:]/128) #Get A and B values

    image8bit = cv2.normalize(LAB_img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    segments = slic(image8bit,n_segments=10,sigma=5,start_label=1)    
    for segmentValue in np.unique(segments):    
        mask = np.zeros(image8bit.shape[:2],dtype='uint8')
        mask[segments == segmentValue] = 255
        #---------------------------------------------SIFT---------------------------------------------------------#
        features,descriptions = sift.detectAndCompute(cv2.bitwise_and(image8bit, image8bit, mask = mask),None)
        sift_im = cv2.drawKeypoints(cv2.bitwise_and(image8bit,image8bit, mask = mask),features,None)
        SIFT_Features.append(sift_im)
        #---------------------------------------------GABOR--------------------------------------------------------#
        filt_img = cv2.filter2D(cv2.bitwise_and(image8bit, image8bit, mask = mask),cv2.CV_8UC3,kernel)
        GABOR_Features.append(filt_img)
           
train_x = np.array(train_x)
train_y = np.array(train_y)

#change dimensions for train_x
train_x = train_x.reshape(train_x.shape+(1,))

#------------------------------------Training Model/Image Classifier----------------------------------#

#---------------------------------------Encoder-------------------------------------------------------#
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256, 256, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

#--------------------------------------Decoder-------------------------------------------------------#
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
model.summary()

model.fit(train_x,train_y,validation_split=0.1, epochs=5, batch_size=16)
model.save('ColorizationModel.model')
