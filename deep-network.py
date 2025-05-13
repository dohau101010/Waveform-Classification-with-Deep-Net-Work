import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
import seaborn as sns

import pathlib
training_dir = pathlib.Path('../input/radardataset/training_set')
training_count = len(list(training_dir.glob('*/*.png')))
print(training_count)

test_dir = pathlib.Path('../input/radardataset/test_set')
test_count = len(list(test_dir.glob('*/*.png')))
print(test_count)

batch_size = 64

img_height = 128
img_width = 128
train_ds = tf.keras.utils.image_dataset_from_directory(
  training_dir,
  validation_split=0,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  validation_split=0,
  seed=113,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


def create_model():
    
    imageinput = tf.keras.Input(shape=(128, 128, 3))
    
    conv = layers.Conv2D(32, (7, 7), padding="same", strides=(2, 2))(imageinput)
    batchnorm = layers.BatchNormalization(epsilon=0.000010)(conv)
    relu = layers.ReLU()(batchnorm)
    
    
    conv11 = layers.Conv2D(16, (1, 1), padding="same", strides=(1, 1))(relu)
    relu111 = layers.ReLU()(conv11)
    group_conv11 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), groups=2, padding='same')(relu111)
    group_conv11 = layers.BatchNormalization(epsilon=0.000010)(group_conv11)
    relu11 = layers.ReLU()(group_conv11)
    
    conv21 = layers.Conv2D(16, (1, 1), padding="same", strides=(1, 1))(relu)
    relu211 = layers.ReLU()(conv21)
    group_conv21 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), groups=2, padding='same')(relu211)
    batchnorm11 = layers.BatchNormalization(epsilon=0.000010)(group_conv21)
    relu21 = layers.ReLU()(batchnorm11)
    
    addition11 = tf.keras.layers.Add()([relu , relu21, relu11])
    pool111 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), padding="same", strides=(1, 1))(addition11)
    print(addition11.shape)
    
    
    
 
    pool31 = tf.keras.layers.AveragePooling2D(pool_size=(1, 32), padding="same", strides=(1, 1))(pool111)
    conv31 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 32),padding="same", strides=(1, 1))(pool31)
    print(conv31.shape)
    pool32 = tf.keras.layers.AveragePooling2D(pool_size=(32, 1), padding="same", strides=(1, 1))(pool111)
    conv32 = tf.keras.layers.Conv2D(filters=4, kernel_size=(32, 1), padding="same", strides=(1, 1))(pool32)
    print(conv32.shape)
    depthcat = tf.keras.layers.Concatenate()([conv31,conv32])
    print(depthcat.shape)
    sigmoid3 = tf.keras.layers.Activation('sigmoid') (depthcat) 
    multiplication31 = tf.keras.layers.Multiply()([depthcat, sigmoid3])
    conv33 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same", strides=(1, 1))(multiplication31)
    multiplication32 = tf.keras.layers.Multiply()([conv33, pool111])
    
    convt2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1),padding="same", strides=(1, 1))(multiplication32)
    relut2 = layers.ReLU()(convt2)
    
 

    conv12 = layers.Conv2D(16, (1, 1), padding="same", strides=(1, 1))(relut2)
    relu121 = layers.ReLU()(conv12)
    group_conv12 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), groups=2, padding='same')(relu121)
    group_conv12 = layers.BatchNormalization(epsilon=0.000010)(group_conv12)
    relu12 = layers.ReLU()(group_conv12)
    
    conv22 = layers.Conv2D(16, (1, 1), padding="same", strides=(1, 1))(relut2)
    relu222 = layers.ReLU()(conv22)
    group_conv22 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), groups=2, padding='same')(relu222)
    batchnorm12 = layers.BatchNormalization(epsilon=0.000010)(group_conv22)
    relu22 = layers.ReLU()(batchnorm12)
    
    addition12 = tf.keras.layers.Add()([relut2 , relu22, relu12])
    pool1111 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), padding="same", strides=(1, 1))(addition12)
 
    
    
    conv111 = layers.Conv2D(16, (1, 1), padding="same", strides=(1, 1))(pool1111)
    relu1111 = layers.ReLU()(conv111)
    group_conv111 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), groups=2, padding='same')(relu1111)
    group_conv111 = layers.BatchNormalization(epsilon=0.000010)(group_conv111)
    relu111 = layers.ReLU()(group_conv111)
    
    conv211 = layers.Conv2D(16, (1, 1), padding="same", strides=(1, 1))(pool1111)
    relu2111 = layers.ReLU()(conv211)
    group_conv211 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), groups=2, padding='same')(relu2111)
    batchnorm111 = layers.BatchNormalization(epsilon=0.000010)(group_conv211)
    relu211 = layers.ReLU()(batchnorm111)
    
    addition111 = tf.keras.layers.Add()([pool1111 , relu211, relu111])
    pool11111 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), padding="same", strides=(1, 1))(addition111)
    print(addition111.shape)
    
    convl = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1),padding="same", strides=(2, 2))(pool1111)
    convl = layers.ReLU()(convl)
    gapool = layers.GlobalAveragePooling2D()(convl)
    fc = layers.Dense(64)(gapool)
#     fc = layers.ReLU()(fc)
    fc1 = layers.Dropout(rate=0.1)(fc)
    fc2 = layers.Dense(8)(fc1)
    softmax = layers.Softmax()(fc2)
    
    
    return tf.keras.Model(inputs=[imageinput], outputs=[softmax])
                          
                          
model = create_model()
model.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto')


# Train the model
epochs=40
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs, shuffle=True, callbacks=learning_rate_reduction)


accuracy = model.evaluate(val_ds)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='black')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='blue')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='black')
plt.plot(epochs_range, val_loss, label='Validation Loss',  color='blue')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Confusion matrix
y_pred = np.argmax(model.predict(val_ds), axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
cm = confusion_matrix(y_true, y_pred)
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(10, 8))
sns.heatmap(cm_perc, annot=True, cmap='Greens', fmt='.1f', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (in %)')
plt.show()
