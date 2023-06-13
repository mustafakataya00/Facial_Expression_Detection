import pandas as pd
import numpy as np
import cv2
import os

data = pd.read_csv('fer2013.csv')

train_data = data[data['Usage']=='Training'] #extract rows with usage = training
validation_data = data[data['Usage']=='PublicTest'] #extract rows with usage = PublicTest for validation

#display
print(validation_data.head())
print(train_data.head())

# Split Data into Train / Validation 
# convert pixels to .jpg then classify each one in training folder or validation folder

for index, row in train_data.iterrows():
    pixels = row['pixels'].split(' ')
    img = np.array(pixels, dtype='uint8').reshape((48, 48))
    label = row['emotion']
    folder_name = 'train/' + str(label)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = folder_name + '/' + str(index) + '.jpg'
    cv2.imwrite(file_name, img)
    
for index, row in validation_data.iterrows():
    pixels = row['pixels'].split(' ')
    img = np.array(pixels, dtype='uint8').reshape((48, 48))
    label = row['emotion']
    folder_name = 'validation/' + str(label)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = folder_name + '/' + str(index) + '.jpg'
    cv2.imwrite(file_name, img)
