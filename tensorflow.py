import tensorflow as tf

from tensorflow.keraas import models, layers, dataset

from tensorflow.keraas.dataset import cifar10

from tensorflow.keraas.utils import categorical

# Preprocessing Data

(train_images, train_labels), (test_images, test_labels) = cifar10.load()

# Chaning the Images into binaries and Labels into categories

(train_images, test_images) = train_images/255, test_images/255

(train_labels, test_labels) = categorical(train_labels), categorical(test_labels)

# CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = "relu"))
model.add(layers.MaxPolling2D(3,3))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPooling2D(3,3))

# Compiling the Model
model.compile(optimizer = "adam", loss = "categorical", matrices = ["accuracy"])
# Training the models
model.fit((train_images, test_images), validation_data = (test_images, test_labels))
# Evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Accuracy is {test_accuracy}')




