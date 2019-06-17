import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
Mnist=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=Mnist.load_data()
class_names=['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','nine']
print(train_images.shape())
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
train_images=train_images/255.0
test_images=test_images/255.0
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
#defining the model:
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
     ])
#compiling the model:
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentroy',
              metrics=['accuracy'])
#fitting the model:
model.fit(train_images,train_labels,epochs=5)
test_loss,test_acc=model.evaluate(test_images,test_labels)
print('Test accuracy:',test_acc)
predictions=model.predict(test_images)
print(predictions[0])
x=np.argmax(predictions[0])
print(x)

