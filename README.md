# Automatic-Colorization-System of Black and White Images

This project implements an automatic image colorization system using machine learning and Python 3.7.4. A CNN model is trained on image datasets obtained from Kaggle to predict color information and transform grayscale images into color images.

The project uses the LAB color space, SLIC superpixel segmentation, and texture feature extraction techniques such as Gabor filters and SIFT. The CNN architecture consists of an encoder-decoder structure, which learns image features and predicts the A and B color channels.

The final model is used to colorize grayscale input images, producing RGB images that can be visualized alongside the extracted texture features.

**Technologies:** Python

**Dataset:** https://www.kaggle.com/prasunroy/natural-images
