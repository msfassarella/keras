#usando o modelo salvo pelo keras
import keras
from keras.models import load_model
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

model_loc = 'saved_models/keras_cifar10_trained_model.h5'
index_of_image = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = load_model(model_loc)

print(x_train[0].shape)
#copy the image from data train with shape (1,32,32,3) -> only for test
image_to_test = x_train[index_of_image:index_of_image+1] 
print(image_to_test.shape)

# what class this image is from
image_class = y_train[index_of_image]


# ploting the image under test
# resize it to a lower size to understand the image. Original image is 32x32x3 
#plt.imshow(image_to_test[0])
#plt.show()

# predict
predict = model.predict(image_to_test)
posMaxValue = np.where(predict == np.amax(predict))

#compare prediction class with the real one
print("class = " + str(image_class))
print("predict = " + str(posMaxValue[1][0]))
