[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Buol6fpg)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=16915390)
## Team Name: 
ai-hardware-project-6501-group7

## Team Members:
- Ezeuko Emmanuel
- Jazzmin Poitier

## Project Title:

Age based recommendation system using Brainchip Akida (Neuromorphic architecture)

## Project Description:
This project aims to develop an AI-powered age-based recommendation system leveraging Brainchip Akida, 
a neuromorphic processor designed for energy-efficient and hardware-optimized inference. 
The system uses age estimation from facial images as a key input to recommend personalized 
content, products, or services tailored to specific age groups.

## Key Objectives:
Accurate Age Prediction

    Develop a reliable age estimation model using facial images.
    Ensure the model achieves high accuracy with low Mean Absolute Error (MAE) across different age groups, considering diversity in demographics (gender, ethnicity, etc.).

Energy-Efficient Inference

    Leverage the Brainchip Akida neuromorphic architecture for low-latency, low-power inference.
    Demonstrate the Akida model’s comparable performance to traditional models while optimizing for edge devices.

Seamless Model Conversion

    Implement quantization-aware training (QAT) and convert the trained model to Akida-compatible format using the cnn2snn library.
    Validate the performance of the Akida model in comparison to the original Keras model.

Integration into a Recommendation System

    Use age predictions to design an age-based recommendation system that can suggest tailored products, content, or services.
    Create recommendation rules or algorithms that adapt dynamically to user-specific age ranges.

Real-Time Edge Deployment

    Deploy the solution on an edge device, demonstrating real-time age prediction and recommendation generation.
    Showcase the advantages of Akida’s energy efficiency and speed in resource-constrained environments.

Scalability and Adaptability

    Design the system to scale across diverse datasets, enabling the integration of additional demographic or contextual data for enhanced personalization.
    Adapt the recommendation system to various industries, such as healthcare, retail, or entertainment.

## Technology Stack:
1. Programming language -> python
2. OpenCv for real time image capture
3. UTKFace dataset, akida_models, cnn2snn
4. Keras, tensorflow
5. matplotlib, numpy

## Expected Outcomes:
Accurate and Robust Age Estimation System

    A trained age estimation model capable of predicting ages from facial images with high accuracy, validated using metrics like Mean Absolute Error (MAE).
    The system should perform well across diverse demographic groups (age, gender, ethnicity) present in the UTKFace dataset.

Efficient Neuromorphic Inference with Akida

    A fully functional age estimation model converted to the Brainchip Akida neuromorphic architecture.
    Demonstration of low power consumption, low latency, and comparable or better performance compared to traditional AI models.

Age-Based Recommendation System

    A prototype recommendation engine that dynamically suggests personalized content, services, or products based on predicted age. Examples:
        Healthcare: Suggestions for age-appropriate vitamins or check-ups.
        Retail: Product recommendations for kids, teens, adults, or seniors.
        Entertainment: Customized movie or music suggestions by age group.

Real-Time Deployment on Edge Devices

    A deployed system capable of performing real-time age predictions and recommendations on edge devices, demonstrating the practicality of Akida for real-world applications.
    Example: A kiosk, tablet, or embedded system showcasing the functionality.

Energy Efficiency and Scalability

    Validation of Akida’s energy-efficient performance, making the solution ideal for battery-operated or IoT devices.
    A scalable framework that allows integration of additional features like gender or emotion recognition to enhance recommendations.

Improved User Experience

    The recommendation system should enhance user satisfaction by delivering highly relevant, personalized suggestions.
    Applications spanning industries such as retail, healthcare, entertainment, and education.

Showcase of Neuromorphic Benefits

    Demonstrate the practical advantages of Akida’s neuromorphic architecture, including reduced hardware costs, increased inference speed, and suitability for real-time edge AI applications.






## Introduction

The advancements in AI and machine learning have driven significant progress in computer vision, particularly in facial analysis tasks like age estimation. This task has diverse applications in security, healthcare, retail, and entertainment, enabling targeted marketing, age-specific content recommendations, and healthcare support for age-related conditions. However, scaling these solutions for edge devices presents challenges in computational efficiency, power consumption, and latency.

Neuromorphic computing, exemplified by BrainChip's Akida platform, offers a solution by mimicking the brain's energy-efficient, event-driven processing. Akida's architecture uses spiking neural networks (SNNs) to enable real-time, low-power operations, making it ideal for edge applications.

This project evaluates Akida's potential for age estimation using the UTKFace dataset. Key steps include:

    Training a VGG-based deep learning model in TensorFlow/Keras.
    Quantizing the model to optimize for deployment.
    Converting the quantized model to an Akida-compatible format with the CNN-to-SNN framework.
    Comparing Keras and Akida models in terms of Mean Absolute Error (MAE) and inference efficiency.
    Testing the models on custom images to assess generalization.

The Akida platform’s ability to convert traditional models into spiking formats allows for seamless integration into AI workflows, offering accuracy and efficiency in resource-constrained environments. The study highlights the potential of neuromorphic computing to complement traditional AI, addressing real-time performance and low-power needs for edge deployments.





## Background

Age prediction from facial images is challenging due to variability in features influenced by ethnicity, lifestyle, lighting, and image quality. While convolutional neural networks (CNNs) like VGGNet excel in capturing these complexities, deploying such models on edge devices demands a balance between accuracy, computational efficiency, and power consumption.

The UTKFace dataset, with diverse facial images labeled by age, gender, and ethnicity, serves as a strong foundation for model development. This project evaluates a pre-trained VGG-based model in its native TensorFlow/Keras form as a baseline, followed by quantization and conversion into an Akida-compatible format using the CNN-to-SNN framework. Akida’s spike-based neuromorphic principles enable ultra-low-power operations with competitive accuracy.

The study compares the performance of Keras and Akida models in terms of mean absolute error (MAE) and generalization to unseen data. By also testing predictions on custom images, the project explores the practical viability of neuromorphic computing for real-world edge AI applications, emphasizing trade-offs between accuracy and efficiency.

## How to Test the Code and Set Up the Akida Development Environment

The Akida Development Environment is called **MetaTF**.

MetaTF consists of four Python packages, leveraging the TensorFlow framework, which can be installed via pip from the PyPI repository.

### MetaTF Packages

1. **Model Zoo (`akida-models`)**:  
   - Provides pre-trained, quantized models.
   - Allows easy instantiation and training of Akida-compatible models.

2. **Quantization Tool (`quantizeml`)**:  
   - Enables quantization of CNN and Vision Transformer models.
   - Uses low-bitwidth weights and outputs for optimized performance.

3. **Conversion Tool (`cnn2snn`)**:  
   - Converts models to a binary format for execution on the Akida platform.

4. **Akida Neuromorphic Processor Interface (`akida`)**:  
   - Includes runtime, Hardware Abstraction Layer (HAL), and a software backend.
   - Simulates the Akida Neuromorphic Processor and supports the AKD1000 reference SoC.

---

### Installation

#### Prerequisites
- Python 3.11 or later.
- It is recommended to use a virtual environment (e.g., Conda) for better dependency management.

#### Setting Up the Environment
1. Create and activate a virtual environment using Conda:
   ```bash
   conda create --name akida_env python=3.11
   conda activate akida_env

2. Install the required MetaTF packages using pip:

```bash
pip install akida==2.11.0
pip install cnn2snn==2.11.0
pip install akida-models==1.6.2

3. Alternatively, you can use the requirements.txt file provided in this repository:

    Download the requirements.txt file.
    Run the following command in the folder containing the requirements.txt file:

    ```pip install -r requirements.txt

##  Key Steps in the Code:

    Dataset Loading:
        load_data function loads the training and testing data (e.g., images and corresponding ages).

    Keras Model Evaluation:
        Downloads and loads a pre-trained VGG model (vgg_utk_face.h5).
        Evaluates its performance on the test set using Mean Absolute Error (MAE).

    Quantized Keras Model Evaluation:
        Loads a quantized version of the VGG model from the Akida models library.
        Evaluates its performance on the test set.

    Akida Model Conversion and Evaluation:
        Converts the quantized Keras model to an Akida-compatible model using cnn2snn.convert.
        Predicts using the Akida model and compares its performance (MAE) with the original Keras model.

    Age Prediction for a Custom Image:
        Loads a custom image (img.png), preprocesses it to fit the input dimensions of the models, and predicts the age using both the Keras and Akida models.
        Displays the image alongside predictions.

    Single Test Image Visualization:
        Selects a random test image from the dataset, visualizes it, and compares predictions with the actual age.







## Methodology
This project demonstrates the deployment of a facial age estimation system by leveraging deep learning and neuromorphic computing. Below is a step-by-step explanation of how the code works:
we start with the image capture then before image processing (video.py):

The code captures live video from the default camera and records it to a file named output.mp4 while displaying the video feed in a window. Here's a summary of its functionality:

    Camera Setup:
        Opens the default camera using OpenCV's cv2.VideoCapture.
        Retrieves the default frame dimensions (frame_width and frame_height).

    Video Recording:
        Initializes a VideoWriter object with the codec 'mp4v' to save the video feed to output.mp4 at 20 frames per second.

    Live Capture and Display:
        Continuously captures frames from the camera.
        Writes each frame to the video file.
        Displays the live video feed in a window titled "Camera."

    Interactivity:
        If the user presses the 't' key, it triggers the start_test() function from the imported test module.
        If the 's' key is pressed, the program exits the loop.

    Cleanup:
        Releases the camera and video writer resources.
        Closes the display window to ensure proper cleanup.

This script combines live video recording, real-time display, and interactive functionality for custom testing.

part 2: image processing

1. Data Loading

The code begins by importing the UTKFace dataset, which contains labeled facial images for tasks like age estimation.

    The dataset is loaded using load_data, splitting it into training (x_train, y_train) and testing (x_test, y_test) sets.
    For Akida compatibility, test images are converted to uint8 format to align with the requirements of event-driven neuromorphic computation.

2. Baseline Model: Native Keras

A pre-trained VGG-based model serves as the baseline:

    The VGG model is a simplified version of the original VGGNet, featuring convolutional and pooling layers followed by dense layers for regression.
    The model is loaded using TensorFlow/Keras and compiled with the Adam optimizer and Mean Absolute Error (MAE) as the loss function.
    The model's performance is evaluated on the test set, providing an initial MAE score as a benchmark for comparison.

3. Quantized Keras Model

To optimize the baseline model for edge deployment:

    A pre-trained quantized Keras model is loaded. This model uses 4-bit weights and activations to reduce memory and computational requirements.
    After compiling the quantized model, its MAE is evaluated on the test set, demonstrating the trade-off between reduced precision and accuracy.

4. Akida Conversion

The quantized model is converted to an Akida-compatible format using the cnn2snn library:

    The Akida model processes input data through spike-based neuromorphic principles, achieving ultra-low-power performance.
    The Akida model's MAE is computed and compared to the baseline and quantized Keras models, ensuring performance consistency.

5. Testing on Custom Images

The code demonstrates practical usability by estimating the age of a custom image:

    The image is preprocessed to match the input requirements of both Keras and Akida models.
    Predictions from both models are compared, showcasing their capabilities and generalization.

6. Learning Rate Optimization

The code includes a learning rate scheduler to enhance model training:

    A custom lr_schedule function dynamically adjusts the learning rate based on the training epoch.
    This ensures efficient convergence by applying higher learning rates initially and reducing them as training progresses.

7. Akida Advantages

The Akida model highlights the following benefits:

    Energy Efficiency: Neuromorphic processing enables real-time inference with minimal power consumption, ideal for edge devices.
    Hardware Optimization: The Akida platform is tailored for neuromorphic hardware, leveraging event-driven computation.
    Comparable Accuracy: The model achieves similar performance to traditional methods while being more resource-efficient.

By combining the VGG architecture, quantization techniques, and neuromorphic computing, the code effectively balances accuracy, efficiency, and real-world deployment potential, making it a suitable solution for edge AI applications like age prediction.







## Contributions

1. Some of our contributions involves implementing a learning rate schedule using a callback and 
integrating it into the training process for both the Keras and quantized Keras models.

here we added a learning_rate_schedule function to dynamically adjust the learning rate 
during training.Integrated the LearningRateScheduler callback into the fit methods of both models.
we also trained both the native Keras and quantized Keras models with the learning rate 
        optimization to improve convergence and stability.
        Ensured that the learning rate adjustment does not affect the Akida conversion 
        and evaluation process.

This approach optimizes the training process, potentially improving model performance and
 convergence speed.

2.  Grid search for hyperparameter optimization: this can significantly improve the model's 
performance by systematically exploring the hyperparameter space such as such as 
 learning rate, batch size, and number of epochs.

3. Incorporating data augmentation can significantly enhance the model's robustness by exposing it to varied transformations of the training data
          Added ImageDataGenerator with:
            Rotation: Up to 20 degrees.
            Horizontal Flipping: Random flips.
            Brightness Adjustment: Between 80% to 120% of the original brightness.
            Zoom: Random zoom up to 20%.
        Applied the data augmentation to the training data.

    Grid Search with Augmentation:
        Integrated the ImageDataGenerator into the grid search using the .flow() method for on-the-fly data generation and augmentation during training.

    Validation with Augmented Data:
        Used augmented training data for both the native Keras and quantized Keras models.

    Robustness:
        Enhanced the model's generalization ability by training it on augmented data, ensuring better performance on unseen data.

Benefits:

    Increased Model Robustness: Augmented data helps the model generalize better.
    Enhanced Performance: Grid search ensures the best hyperparameters are used for the augmented dataset.
    End-to-End Optimization: Combines hyperparameter tuning and data augmentation seamlessly.

4. Optimizng for edge deployment using quantization: Optimizing the code for edge deployment involves focusing on efficiency, particularly through model quantization.
    Quantization reduces the precision of weights and activations, enabling faster inference and reduced power consumption without significantly compromising accuracy.
    Here's how the code is modified for edge deployment using quantization
       Quantized Keras Model:
        Used a pre-trained quantized model (vgg_utk_face_pretrained) with 4-bit weights and activations to reduce computational overhead.

    Conversion to Akida Model:
        The cnn2snn.convert() function converts the quantized Keras model into an Akida model suitable for edge hardware. Akida models operate efficiently with low precision (8-bit or lower).

    Reduced Input Precision:
        Converted input images to uint8 format for compatibility with the Akida model, further optimizing memory usage and inference time.

    Lightweight Optimizer:
        Chose the Adam optimizer with a small learning rate for fine-tuning to balance convergence and stability.

    Data Augmentation:
        Applied data augmentation to improve the model’s ability to generalize, which is critical for real-world scenarios encountered during edge deployment.

    Edge-Friendly Architecture:
        Used a quantized VGG-inspired architecture, which is simpler and faster for inference on resource-constrained edge devices.

Benefits of These Changes:

    Memory Efficiency: Quantization and uint8 inputs drastically reduce memory usage.
    Faster Inference: Akida models are optimized for low-power, real-time edge devices.
    Scalability: The approach can be easily adapted to different edge hardware platforms.
    Real-World Robustness: Data augmentation ensures the model generalizes well across diverse scenarios.
5. Pruning: this is a technique that reduces the size of a neural network by removing unnecessary weights, leading to smaller models with reduced inference latency and memory usage.
       Pruning the Quantized Keras Model:
        Used prune_low_magnitude from TensorFlow Model Optimization Toolkit to prune weights with low magnitude.
        Applied constant sparsity with 50% sparsity (half of the weights are set to zero).

    Stripping Pruning Wrappers:
        After training, used strip_pruning to remove pruning-specific wrappers and prepare the model for conversion and deployment.

    Quantization-Aware Pruning:
        Pruning was applied to the already quantized model to ensure compatibility with edge deployment requirements.

    Edge Deployment with Akida:
        Converted the pruned model to the Akida format for efficient inference on edge devices.

Benefits of Pruning for Edge Deployment:

    Reduced Model Size: Pruning eliminates redundant weights, reducing the model's memory footprint.
    Improved Inference Speed: Sparse models can often be processed more quickly on specialized hardware.
    Lower Power Consumption: Smaller models consume less power, which is critical for edge devices.
    Maintained Accuracy: Pruning retains most of the model's accuracy while significantly optimizing resource usage.
