"""
Age estimation (regression) example
==================================================

This tutorial aims to demonstrate the comparable accuracy of the Akida-compatible
model to the traditional Keras model in performing an age estimation task.

It uses the `UTKFace dataset <https://susanqq.github.io/UTKFace/>`__, which
includes image s of faces and age labels, to showcase how well akida compatible
model can predict the ages of individuals based on their facial features.

"""

######################################################################
# 1. Load the UTKFace Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The UTKFace dataset has 20,000+ diverse face images spanning 0 to 116 years.
# It includes age, gender, ethnicity annotations. This dataset is useful for
# various tasks like age estimation, face detection, and more.
#
# Load the dataset from Brainchip data server using the `load_data
# <../../api_reference/akida_models_apis.html#akida_models.utk_face.preprocessing.load_data>`__
# helper (decode JPEG images and load the associated labels).

from rec import *
from akida_models.utk_face.preprocessing import load_data

def start_test():
  print("Load the dataset")
  x_train, y_train, x_test, y_test = load_data()

  ######################################################################
  # Akida models accept only `uint8 tensors <../../api_reference/akida_apis.html?highlight=uint8#akida.Model>`_
  # as inputs. Use uint8 raw data for Akida performance evaluation.

  # For Akida inference, use uint8 raw data
  x_test_akida = x_test.astype('uint8')


  ######################################################################
  print(" 2. Load a pre-trained native Keras model")
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #
  # The model is a simplified version inspired from `VGG <https://arxiv.org/abs/1409.1556>`__
  # architecture. It consists of a succession of convolutional and pooling layers
  # and ends with two dense layers that outputs a single value
  # corresponding to the estimated age.
  #
  # The performance of the model is evaluated using the "Mean Absolute Error"
  # (MAE). The MAE, used as a metric in regression problem, is calculated as an
  # average of absolute differences between the target values and the predictions.
  # The MAE is a linear score, i.e. all the individual differences are equally
  # weighted in the average.

  from akida_models import fetch_file
  from tensorflow.keras.models import load_model

  print(" Retrieve the model file from the BrainChip data server")
  model_file = fetch_file(fname="vgg_utk_face.h5",
                          origin="https://data.brainchip.com/models/AkidaV2/vgg/vgg_utk_face.h5",
                          cache_subdir='models')

  print(" Load the native Keras pre-trained model")
  model_keras = load_model(model_file)
  model_keras.summary()

  ######################################################################

  print(" Compile the native Keras model (required to evaluate the MAE)")
  model_keras.compile(optimizer='Adam', loss='mae')

  print(" Check Keras model performance")
  mae_keras = model_keras.evaluate(x_test, y_test, verbose=0)

  print("Keras MAE: {0:.4f}".format(mae_keras))

  ######################################################################
  print(" 3. Load a pre-trained quantized Keras model")
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #age_akidalayers are quantized using 4-bit weights, all
  # activations are 4-bit.
  #age_akida

  from akida_models import vgg_utk_face_pretrained

  print(" Load the pre-trained quantized model")
  model_quantized_keras = vgg_utk_face_pretrained()
  model_quantized_keras.summary()

  ######################################################################

  print(" Compile the quantized Keras model (required to evaluate the MAE)")
  model_quantized_keras.compile(optimizer='Adam', loss='mae')

  # Check Keras model performance
  mae_quant = model_quantized_keras.evaluate(x_test, y_test, verbose=0)

  print("Keras MAE: {0:.4f}".format(mae_quant))

  ######################################################################
  print(" 4. Conversion to Akida")
  # ~~~~~~~~~~~~~~~~~~~~~~
  #
  print(" The quantized Keras model is now converted into an Akida model. After conversion, we evaluate the")
  print(" performance on the UTKFace dataset.")
  #

  from cnn2snn import convert

  # Convert the model
  model_akida = convert(model_quantized_keras)
  model_akida.summary()

  #####################################################################

  import numpy as np

  print(" Check Akida model performance")
  y_akida = model_akida.predict(x_test_akida)

  print(" Compute and display the MAE")
  mae_akida = np.sum(np.abs(y_test.squeeze() - y_akida.squeeze())) / len(y_test)
  print("Akida MAE: {0:.4f}".format(mae_akida))

  # For non-regression purposes
  assert abs(mae_keras - mae_akida) < 0.5

  ######################################################################
  print(" 5. Estimate age on a single image")
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Select a random image from the test set for age estimation.
  # Print the Keras model's age prediction using the ``model_keras.predict()`` function.
  # Print the Akida model's estimated age and the actual age associated with the image.

  import matplotlib.pyplot as plt
  from tensorflow.keras.preprocessing.image import load_img, img_to_array
  import numpy as np
  your_image_path = "img.png"

  print(" 1. Load and preprocess your image")
  img = load_img(your_image_path, target_size=(32 , 32))  # Resize to match the model input size
  img_array = img_to_array(img)  # Convert to NumPy array
  img_array = img_array / 255.0  # Normalize (if required by the Keras model)
  img_array_akida = (img_array * 255).astype('uint8')  # Prepare for Akida model
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  img_array_akida = np.expand_dims(img_array_akida, axis=0)

  print(" 2. Predict age using Keras model")
  age_keras = model_keras.predict(img_array)

  print(" 3. Predict age using Akida model")
  age_akida = model_akida.predict(img_array_akida)


  print(" Estimate age on a random single image and display Keras and Akida outputs")
  id = np.random.randint(0, len(y_test) + 1)
  #age_keras = model_keras.predict(x_test[id:id + 1])

  plt.imshow(img, interpolation='bicubic')
  plt.xticks([]), plt.yticks([])
  plt.show()

  print("Keras estimated age: {0:.1f}".format(age_keras.squeeze()))
  print("Akida estimated age: {0:.1f}".format(age_akida.squeeze()))
  print(f"Actual age: {y_test[id].squeeze()}")
  get_recommendation(age_akida.squeeze())
