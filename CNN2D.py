import numpy as np 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix
np.set_printoptions(threshold=sys.maxsize)



data=pd.read_csv('Train4040.csv')
#print(data.head(10))


ohe = OneHotEncoder()
y = data['1601'].to_numpy() 
y = y.reshape(len(y),1) 
x = data.drop(['1601'],1).to_numpy()
#X = np.expand_dims(x, axis=2)
Y = ohe.fit_transform(y).toarray() 


########################## Dimension Increase #############################

main = []
for i in range(len(x)):
    d = x[i]
    d = np.reshape(d,(40,40,1))
    main.append(d)
main=np.array(main)
X = main
###########################################################################


############################## Data Print #################################
"""
for i in range(50):
  plt.subplot( 50// 10, 10, i + 1)
  plt.imshow((x[i].reshape(40,40))*255,cmap=plt.cm.gray)
plt.show()
"""
###########################################################################


epochs =30
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.0,random_state=0)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(8, (3,3), activation='relu',input_shape = (40,40,1,)))
model.add(tf.keras.layers.BatchNormalization()) #covariateshift
#model.add(tf.keras.layers.MaxPooling2D(2,2))
#model.add(tf.keras.layers.Dropout(0.2))
    
model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.MaxPooling2D(2,2))
#model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(32,(3,3), activation='relu')) 
model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.MaxPooling2D(2,2))
#model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.MaxPooling2D(2,2))
#model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(800, activation='relu'))
#model.add(tf.keras.layers.Dense(100, activation='relu'))
#model.add(tf.keras.layers.Dense(230, activation='relu'))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
#model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))  
model.summary()
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(
                                   zoom_range = 0.15,
                                   fill_mode = 'nearest') #augmentation
train_generator = train_datagen.flow(x_train,
                                     y_train)
validation_datagen = ImageDataGenerator()
#validation_generator = validation_datagen.flow(x_test,
#                                     y_test)

history = model.fit_generator(train_generator,
                             epochs=epochs)
#                             validation_data=validation_generator)


test_data = pd.read_csv('Test4040+503.csv')
y1 = test_data['1601'].to_numpy() 
#y1 = y1.reshape(1,len(y1)) 
x1 = test_data.drop(['1601'],1).to_numpy()


main1 = []
for i in range(len(x1)):
    d = x1[i]
    d = np.reshape(d,(40,40,1))
    main1.append(d)

main1=np.array(main1)


for i in range(46):
  plt.subplot( 50// 10, 10, i + 1)
  view = x1[i]
  plt.imshow(view.reshape(40,40)*255,cmap=plt.cm.gray)
  plt.axis('off')
  #plt.subplot(1,46,i+1)
  #view = test_x[i+46]
  #plt.imshow((view.reshape(40,40))*255,cmap=plt.cm.gray)
plt.show()


"""
test_x = main1
#test_y = np.append(y1,test_y1)
test_y = y1
y_prob = model.predict(test_x)
predicted_y =np.argmax(y_prob, axis=1)+1

#pd_predicted_y = pd.DataFrame(predicted_y)
#pd_actual_y = pd.DataFrame(test_y)
#pd_predicted_y.to_excel('pd_predicted_y.xlsx',index=False)
#pd_actual_y.to_excel('pd_actual_y.xlsx',index=False)
#c = confusion_matrix(test_y, predicted_y)
#cd = pd.DataFrame(c)
#cd.to_excel('confusion matrix.xlsx',index=False)
accuracy = accuracy_score(test_y,predicted_y)
print(accuracy)
"""

"""
epochs_range = range(1,epochs+1)
plt.plot(epochs_range, history.history['acc'])
#plt.plot(epochs_range ,history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Acurracy')
plt.xlabel('Epochs')
plt.show()

plt.plot(epochs_range,history.history['loss'])
#plt.plot(epochs_range,history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
"""