#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import easygui
import imageio


# In[2]:


from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img


# In[3]:


datagen=ImageDataGenerator(
                rotation_range=45, 
                width_shift_range=0.2, 
                height_shift_range=0.2, 
                shear_range=0.2, 
                zoom_range=0.2, 
                horizontal_flip=True, 
                fill_mode='nearest')


# In[4]:


ImagePath=easygui.fileopenbox()
image=cv2.imread(ImagePath)
if image is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
plt.imshow(image)
plt.axis('off')
plt.show()
x=img_to_array(image)
x=x.reshape((1,)+x.shape)


# In[5]:


i=0
for batch in datagen.flow(x, batch_size=1, save_to_dir=r'C:\Users\Aditi\Desktop\Projects\Shoe detection\Adidas\preview', save_prefix='shoe',save_format='jpeg'):
    i+=1
    if i>19:
        break

