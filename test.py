import os
import numpy as np
import predict
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/NSFW (1).jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

CNN = predict.newCNN()

#========================================================================================
# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale = 1./255,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)
# training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')

# test_datagen = ImageDataGenerator(rescale = 1./255)
# test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')

#========================================================================================
# loss, acc = CNN.evaluate(training_set, test_set, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
CNN.load_weights(checkpoint_path)

# loss, acc = CNN.evaluate(training_set, test_set, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

result = CNN.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'nsfw'
else:
    prediction = 'sfw'

print(prediction)