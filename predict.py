def newCNN():
    import tensorflow as tf
    CNN = tf.keras.models.Sequential()
    CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    CNN.add(tf.keras.layers.Flatten())
    CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))
    CNN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    CNN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return CNN
def predict():
    import numpy as np
    from keras.preprocessing import image
    import os
    test_image = image.load_img(message.attachments[0]['url'], target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    CNN = newCNN()
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    CNN.load_weights(checkpoint_path)
    result = CNN.predict(test_image)
    training_set.class_indices
    if result[0][0] == 1:
        return 'nsfw'
    else:
        return 'sfw'