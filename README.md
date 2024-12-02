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






Introduction

The rapid advancements in artificial intelligence (AI) and machine learning (ML) have enabled significant progress in computer vision, particularly in tasks involving facial analysis. Age estimation from facial images is one such task that has found applications in diverse fields such as security, healthcare, retail, and entertainment. Accurate age prediction models can support applications like targeted marketing, age-appropriate content recommendation, and assistive technologies in healthcare for the early detection of age-related conditions. However, deploying such solutions at scale, particularly on edge devices, requires overcoming challenges related to computational efficiency, power consumption, and latency.

Neuromorphic computing has emerged as a promising paradigm for addressing these challenges. By mimicking the human brain's energy-efficient processing mechanisms, neuromorphic architectures, such as BrainChip's Akida platform, offer a hardware solution that combines high performance with low power consumption. Unlike traditional neural networks that rely on synchronous, frame-based computation, the Akida architecture employs event-driven processing, wherein computations occur only in response to spikes or events. This event-based processing enables Akida models to operate efficiently in real-time, making them well-suited for edge applications.

This project aims to explore the potential of Akida's neuromorphic computing platform for age prediction tasks. Specifically, it evaluates the performance of an age prediction model using the UTK Face dataset—a diverse benchmark dataset for facial analysis. The project involves the following steps:

    Training and evaluating a traditional deep learning model, based on the VGG architecture, in its native TensorFlow/Keras form.
    Quantizing the Keras model for optimized deployment without significant loss of accuracy.
    Converting the quantized model into an Akida-compatible format using the CNN-to-SNN (cnn2snn) framework.
    Comparing the performance of the Keras and Akida models in terms of Mean Absolute Error (MAE) and inference efficiency.
    Testing both models on custom facial images to evaluate their generalization capabilities.

The Akida architecture offers a unique approach to implementing Spiking Neural Networks (SNNs), which are inherently energy-efficient due to their sparse, asynchronous computation. The platform supports both native SNNs and the conversion of pre-trained deep learning models into spiking-compatible formats. This flexibility allows for seamless integration with existing AI workflows while unlocking the benefits of neuromorphic computing.

By leveraging Akida's strengths, this project investigates the trade-offs between accuracy, efficiency, and practical deployment capabilities in resource-constrained environments. The findings aim to provide insights into how neuromorphic computing can complement traditional AI techniques, particularly for tasks like age prediction, where real-time performance and low-power consumption are critical.


Background

Age prediction using facial images is a challenging task due to the high variability in facial features caused by factors such as ethnicity, lifestyle, lighting conditions, and image quality. Deep learning models, particularly convolutional neural networks (CNNs), have been successful in learning the complex mappings between facial features and age labels. However, deploying such models on edge devices often requires balancing accuracy with computational efficiency and power consumption.

The UTK Face dataset provides a robust foundation for developing age prediction models, as it contains a wide variety of facial images labeled with age, gender, and ethnicity. Pre-trained models like VGGNet have shown excellent performance on similar tasks, and quantization techniques can further optimize these models for deployment without significant loss of accuracy.

In this project, a pre-trained VGG-based model is first evaluated in its native TensorFlow/Keras form to establish a baseline performance. This model is then quantized and converted into an Akida-compatible format using the CNN-to-SNN (cnn2snn) framework. Akida models leverage spike-based neuromorphic principles, enabling ultra-low-power execution while maintaining competitive accuracy. This study aims to highlight the trade-offs in accuracy and efficiency between traditional deep learning and neuromorphic approaches.

By assessing the performance of both the Keras and Akida models in terms of mean absolute error (MAE) and their ability to generalize to unseen data, this project aims to provide insights into the viability of neuromorphic computing for real-world edge AI applications. The additional evaluation of predictions on custom images further demonstrates the practical usability of these models in various scenarios.

Key Steps in the Code:

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


Contributions

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
