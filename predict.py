
from akida_models.utk_face.preprocessing import load_data

def start_test():
  # Load the dataset
  x_train, y_train, x_test, y_test = load_data()


  x_test_akida = x_test.astype('uint8')

  from akida_models import fetch_file
  from tensorflow.keras.models import load_model

  # Retrieve the model file from the BrainChip data server
  model_file = fetch_file(fname="vgg_utk_face.h5",
                          origin="https://data.brainchip.com/models/AkidaV2/vgg/vgg_utk_face.h5",
                          cache_subdir='models')

  # Load the native Keras pre-trained model
  model_keras = load_model(model_file)
  model_keras.summary()

  ######################################################################


  model_keras.compile(optimizer='Adam', loss='mae')

  mae_keras = model_keras.evaluate(x_test, y_test, verbose=0)

  print("Keras MAE: {0:.4f}".format(mae_keras))

  from akida_models import vgg_utk_face_pretrained


  model_quantized_keras = vgg_utk_face_pretrained()
  model_quantized_keras.summary()

  model_quantized_keras.compile(optimizer='Adam', loss='mae')


  mae_quant = model_quantized_keras.evaluate(x_test, y_test, verbose=0)

  print("Keras MAE: {0:.4f}".format(mae_quant))

  from cnn2snn import convert

  # Convert the model
  model_akida = convert(model_quantized_keras)
  model_akida.summary()

  #####################################################################

  import numpy as np

  # Check Akida model performance
  y_akida = model_akida.predict(x_test_akida)

  # Compute and display the MAE
  mae_akida = np.sum(np.abs(y_test.squeeze() - y_akida.squeeze())) / len(y_test)
  print("Akida MAE: {0:.4f}".format(mae_akida))

  # For non-regression purposes
  assert abs(mae_keras - mae_akida) < 0.5

  import matplotlib.pyplot as plt
  from tensorflow.keras.preprocessing.image import load_img, img_to_array
  import numpy as np
  your_image_path = "img.png"

  # 1. Load and preprocess your image
  img = load_img(your_image_path, target_size=(32 , 32))  # Resize to match the model input size
  img_array = img_to_array(img)  # Convert to NumPy array
  img_array = img_array / 255.0  # Normalize (if required by the Keras model)
  img_array_akida = (img_array * 255).astype('uint8')  # Prepare for Akida model
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  img_array_akida = np.expand_dims(img_array_akida, axis=0)

  # 2. Predict age using Keras model
  age_keras = model_keras.predict(img_array)

  # 3. Predict age using Akida model
  age_akida = model_akida.predict(img_array_akida)


  # Estimate age on a random single image and display Keras and Akida outputs
  id = np.random.randint(0, len(y_test) + 1)
  #age_keras = model_keras.predict(x_test[id:id + 1])

  plt.imshow(img, interpolation='bicubic')
  plt.xticks([]), plt.yticks([])
  plt.show()

  print("Keras estimated age: {0:.1f}".format(age_keras.squeeze()))
  print("Akida estimated age: {0:.1f}".format(age_akida.squeeze()))
  print(f"Actual age: {y_test[id].squeeze()}")
