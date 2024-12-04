from tensorflow.keras.optimizers import Adam
from akida_models import vgg_utk_face_pretrained
from cnn2snn import convert
from akida import Model as AkidaModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def start_test():
    ######################################################################
    # 1. Load the Dataset
    ######################################################################
    print("Loading UTKFace dataset...")
    from akida_models.utk_face.preprocessing import load_data
    x_train, y_train, x_test, y_test = load_data()

    ######################################################################
    # 2. Data Augmentation
    ######################################################################
    print("Applying data augmentation...")
    data_gen = ImageDataGenerator(
        rotation_range=20,           # Random rotation in degrees
        horizontal_flip=True,       # Random horizontal flipping
        brightness_range=[0.8, 1.2], # Random brightness adjustment
        zoom_range=0.2,             # Random zoom
        fill_mode='nearest'         # Filling strategy for transformed pixels
    )

    # Fit the generator to the training data
    data_gen.fit(x_train)

    ######################################################################
    # 3. Load and Fine-Tune Quantized Keras Model
    ######################################################################
    print("Loading and fine-tuning the quantized Keras model...")
    model_quantized_keras = vgg_utk_face_pretrained()

    # Compile the model with an edge-deployment-friendly optimizer
    model_quantized_keras.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mae'])

    # Train the model with augmented data
    model_quantized_keras.fit(
        data_gen.flow(x_train, y_train, batch_size=32),
        validation_data=(x_test, y_test),
        epochs=10
    )

    # Evaluate the quantized Keras model
    mae_quant = model_quantized_keras.evaluate(x_test, y_test, verbose=0)
    print(f"Quantized Keras MAE: {mae_quant:.4f}")

    ######################################################################
    # 4. Convert to Akida Model for Edge Deployment
    ######################################################################
    print("Converting the quantized Keras model to Akida for edge deployment...")
    model_akida = convert(model_quantized_keras)
    model_akida.summary()

    # Evaluate the Akida model
    x_test_akida = x_test.astype('uint8')  # Akida models require uint8 inputs
    y_akida = model_akida.predict(x_test_akida)
    mae_akida = np.sum(np.abs(y_test.squeeze() - y_akida.squeeze())) / len(y_test)
    print(f"Akida MAE: {mae_akida:.4f}")

    ######################################################################
    # 5. Single Image Prediction for Edge Testing
    ######################################################################
    print("Running single image prediction...")
    your_image_path = "img.png"  # Path to a test image
    img = load_img(your_image_path, target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array_akida = (img_array * 255).astype('uint8')
    img_array = np.expand_dims(img_array, axis=0)
    img_array_akida = np.expand_dims(img_array_akida, axis=0)

    age_keras = model_quantized_keras.predict(img_array)
    age_akida = model_akida.predict(img_array_akida)

    plt.imshow(img, interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()

    print(f"Keras estimated age: {age_keras.squeeze():.1f}")
    print(f"Akida estimated age: {age_akida.squeeze():.1f}")

# Run the test
start_test()
