''' Imports '''
import get_images
import get_landmarks
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import cv2

cap = cv2.VideoCapture('../Desktop/Vids/Task#1.mp4')
i=0
f=1 

path = '../Desktop/Vids/F20DataImages/C_Calderon'
os.makedirs(path)


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if f%100 == 0:
        cv2.imwrite(os.path.join(path, 'Task1-Frame'+str(i)+'.png'),frame)
    i+=1
    f+=1
cap.release()

print("FINISHED")

"""
''' Load the data and their labels '''
image_directory = 'Caltech Faces Dataset'
X, y = get_images.get_images(image_directory)

''' Get distances between face landmarks in the images '''
# get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False)
X, y = get_landmarks.get_landmarks(X, y, 'landmarks/', 68, False)


''' kNN classification treating every sample as a query'''
# initialize the classifier

knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean') 
num_correct = 0
labels_correct = []
num_incorrect = 0
labels_incorrect = []

for i in range(0, len(y)):
    query_img = X[i, :]
    query_label = y[i]
    
    template_imgs = np.delete(X, i, 0)
    template_labels = np.delete(y, i)
        
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_labels))
    y_hat[template_labels == query_label] = 1 
    y_hat[template_labels != query_label] = 0
    
    knn.fit(template_imgs, y_hat) # Train the classifier
    y_pred = knn.predict(query_img.reshape(1,-1)) # Predict the label of the query
    
    # Print results
    if y_pred == 1:
        num_correct += 1
        labels_correct.append(query_label)
    else:
        num_incorrect += 1
        labels_incorrect.append(query_label)

# Print results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))    
    
    """