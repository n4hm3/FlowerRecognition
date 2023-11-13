from utils import load_data
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

(feature, labels) = load_data()

x_train, x_test, y_train, y_test = train_test_split(
    feature, labels, test_size=0.1)

cat = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = tf.keras.models.load_model('mod.h5')

pred = model.predict(x_test)
print(pred.shape)
for i in range(20, 30):
    img = cv2.cvtColor(x_test[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xlabel('Actual:'+cat[y_test[i]]+'\n'+'Prediction: '
               + cat[np.argmax(pred[i])])
    plt.xticks([])
    plt.show()
