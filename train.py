from keras.applications import MobileNet
from keras.models import Sequential,Model 
from keras.layers import Dense,Dropout,Activation,Flatten,GlobalAveragePooling2D
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Multiply
import os

from keras.preprocessing.image import ImageDataGenerator

# MobileNet is designed to work with images of dim 224,224
img_rows,img_cols = 224,224

# We load the MobileNet pre-trained model from the Keras applications module. 
# weights='imagenet' means that we will use the pre-trained weights on the ImageNet dataset. 
# include_top=False means that we will exclude the last fully connected layer (top layer) of the model:
MobileNet = MobileNet(weights='imagenet',include_top=False,input_shape=(img_rows,img_cols,3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True
# This means that when we train our model, 
# these layers will also be updated with new weights
for layer in MobileNet.layers:
    layer.trainable = True

# Let's print our layers
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i),layer.__class__.__name__,layer.trainable)

#creates the top or head of the model that will be placed ontop of the bottom layers
def addTopModelMobileNet(bottom_model, num_classes):

    top_model = bottom_model.output #taking the output from the last layer of the bottom_model as the input for the top model.
    
    top_model = GlobalAveragePooling2D()(top_model) #calculates average for each channel // pooling layer
    
    top_model = Dense(1024,activation='relu')(top_model) #ReLU is preferred over other activation functions because it is computationally efficient 
                                                         #and has been shown to improve the accuracy of deep neural networks.
    
    # Add attention mechanism
    attention = Conv2D(1, (1,1), padding='same', activation='sigmoid')(bottom_model.output) #The output of the sigmoid activation function is multiplied 
    #element-wise with the output of the bottom_model. 
    #This allows the model to focus more on the important regions of the input image.
    attention = Multiply()([attention, bottom_model.output])
    attention = GlobalAveragePooling2D()(attention) #// pooling layer
    
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    
    # give more weightage to certain regions in the feature maps.
    # We first multiply the output of the global average pooling layer with the attention output. This helps in emphasizing the relevant features and suppress the irrelevant ones
    top_model = Multiply()([top_model, attention])
    
    # obtain the final output probabilities for each class.
    # This layer takes in the concatenated output from the previous layer and maps it to a probability distribution over the output classes.
    top_model = Dense(num_classes,activation='softmax')(top_model)

    return top_model


num_classes = 7

# create cus the final model by combining the pre-trained MobileNet model with thestom top layers
FC_Head = addTopModelMobileNet(MobileNet, num_classes)

#creates a new model that can be trained to classify images into the 7 different categories.
model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())

train_data_dir = 'C:/Users/user/Desktop/Ai_Project/Emotion-Detection/train'
validation_data_dir = 'C:/Users/user/Desktop/Ai_Project/Emotion-Detection/validation'

#Here, the ImageDataGenerator instance is used to create train and validation data generators, which generate batches of data from image directories.
# The rescale argument in the ImageDataGenerator scales the pixel values of the images to be in the range of [0,1], 
# which is necessary for the training of neural networks.
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,
                    width_shift_range=0.3,
                    height_shift_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest'
                                   )

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32 #This is the number of images that will be processed at once during training.


# method to load the training and validation data from their respective directories 
# .flow_from_directory generates batches of augmented images and their corresponding labels on-the-fly.

train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size = (img_rows,img_cols),# We specify the target image size,batch size
                        batch_size = batch_size,
                        class_mode = 'categorical' #class mode as categorical (since we have multiple classes)
                        )

validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            target_size=(img_rows,img_cols),
                            batch_size=batch_size,
                            class_mode='categorical')

from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

#ModelCheckpoint saves the model with the best validation loss. 
checkpoint = ModelCheckpoint(
                             'emotion_face_mobilNet.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

#stops training early if there is no improvement in the validation loss
earlystop = EarlyStopping(
                          monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)

#ReduceLROnPlateau reduces the learning rate if the validation accuracy does not improve after a certain number of epochs
#This can help the optimizer to converge to the optimal solution more effectively.
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=5, # This means that if the validation accuracy does not improve for 5 epochs,  
                                            verbose=1, #Progress bar mode, a progress bar is displayed for each epoch during training.
                                            factor=0.2, #the learning rate will be reduced by a factor of 0.2.
                                            min_lr=0.0001 #The minimum learning rate is set to 0.0001 to ensure that the learning rate does not become too small.
)

callbacks = [earlystop,checkpoint,learning_rate_reduction] #define callbacks

model.compile(loss='categorical_crossentropy', #commonly used for multi-class classification problems
              optimizer=Adam(lr=0.001), #We are using the Adam optimizer with a learning rate of 0.001
              metrics=['accuracy'] #We want to evaluate the performance of the model using the accuracy metric. 
                                   #The accuracy metric measures the fraction of images that are classified correctly.
              )

nb_train_samples = 24176
nb_validation_samples = 3006

epochs = 25

history = model.fit(
            train_generator,
            steps_per_epoch=nb_train_samples//batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples//batch_size)

model.save('Emotion_Detection.h5')

