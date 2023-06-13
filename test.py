#Keras is used for loading the trained deep learning model,
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from tensorflow.keras.preprocessing import image
import cv2 #OpenCV is used for detecting faces and visualizing the prediction results
import numpy as np #NumPy is used for numerical computation

face_classifier = cv2.CascadeClassifier('C:/Users/user/Desktop/Ai_Project/Emotion-Detection/haarcascade_frontalface_default.xml')
classifier =load_model('C:/Users/user/Desktop/Ai_Project/Emotion-Detection/Emotion_Detection.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise'] 

cap = cv2.VideoCapture(0) #open video cam

total_correct = 0
total_samples = 0

while True:
    # Grab a single frame of video
    ret, frame = cap.read() #ret is to store the status of frame capture
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # convert frame to grayscale, which improves the efficiency of the face detection process.
    faces = face_classifier.detectMultiScale(gray,1.3,5) #detect faces in the grayscale image using the Haar Cascade Classifier
                                                         #usually , detectMultiScale returns the coordinates of the detected faces in the image.

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #draw rectangle around the detected face
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA) #face region cropped from grayscale resized to 48x48 pixels 
                                                                             #that's what is called roi (region of interest)


        if np.sum([roi_gray])!=0: #checking if roi isn't empty , then face detected
            roi = roi_gray.astype('float')/255.0 #converts pixel values to values btw 0-1
            roi = img_to_array(roi) #converts the image to a NumPy array.
            roi = np.expand_dims(roi,axis=0) #adds an extra dimension to the array, which is required by the model to represent a batch of images. 
                                             #The axis=0 argument specifies that the extra dimension should be added as the first dimension.

    # make a prediction on the ROI, then lookup the class
    preds = classifier.predict(roi)[0] #makes a prediction on the ROI using the trained model. 
                                       #The predict() method returns an array of probabilities for each class. 
                                       # The [0] index is used to extract the prediction for the single input image.
    
    print("\nprediction = ",preds)
    
    label = class_labels[preds.argmax()] #checks for the highest probability class label
    print("\nprediction max = ",preds.argmax())
    print("\nlabel = ",label)
    
    label_position = (x,y) #specifies the position where the predicted label should be displayed on the frame
    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3) #adds the predicted label to the frame 
                                                                                    #The FONT_HERSHEY_SIMPLEX argument specifies the font style, 
                                                                                    #2 specifies the font scale, 
                                                                                    #(0,255,0) specifies the font color (in BGR format), 
                                                                                    # 3 specifies the font thickness.
    cv2.imshow('Emotion Detector',frame)#display frame
    
    
    # generate heatmap
    heatmap = cv2.resize(preds, (roi_gray.shape[1], roi_gray.shape[0])) #resizes the predicted probabilities to the size of the roi_gray image. 
                                                                        #This is required to generate a heatmap of the same size as the face ROI
                                                                        
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET) #applies a color map to the heatmap to visualize the intensity of each pixel.
                                                                        #The applyColorMap() function takes two arguments: the grayscale image to be color-mapped, 
                                                                        # the colormap to be used (cv2.COLORMAP_JET in this case).
                                                                        
    heatmap = cv2.resize(heatmap,(frame.shape[1], frame.shape[0]) , interpolation=cv2.INTER_LINEAR) #resizes the heatmap to the size of the original frame.

    # apply heatmap on the original frame
    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)


    
    cv2.imshow('Emotion Detector with Heatmap', overlay) # display the frame with heatmap
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #to exit the program
        break

cap.release()
cv2.destroyAllWindows()














