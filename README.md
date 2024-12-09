
# Team Name  
**ai-hardware-project-6501-group7**

## Team Members  
- **Ezeuko Emmanuel**  
- **Jazzmin Poitier**

## Project Title  
**Age-Based Recommendation System Using BrainChip Akida (Neuromorphic Architecture)**

---

## Project Description  
This project aims to develop an AI-powered age-based recommendation system leveraging BrainChip Akida, a neuromorphic processor designed for energy-efficient and hardware-optimized inference. The system uses age estimation from facial images as a key input to recommend personalized content, products, or services tailored to specific age groups.

---

## Key Objectives  

### Accurate Age Prediction  
- Develop a reliable age estimation model using facial images.  
- Ensure high accuracy with low Mean Absolute Error (MAE) across diverse demographics (gender, ethnicity, etc.).

### Energy-Efficient Inference  
- Leverage the BrainChip Akida neuromorphic architecture for low-latency, low-power inference.  
- Demonstrate Akida’s comparable performance to traditional models while optimizing for edge devices.

### Seamless Model Conversion  
- Implement quantization-aware training (QAT) and convert the trained model to Akida-compatible format using the `cnn2snn` library.  
- Validate the Akida model’s performance compared to the original Keras model.

### Integration into a Recommendation System  
- Use age predictions to design a system suggesting tailored content, products, or services.  
- Create dynamic recommendation algorithms adapting to user-specific age ranges.

### Real-Time Edge Deployment  
- Deploy the solution on an edge device, demonstrating real-time age prediction and recommendation generation.  
- Highlight Akida’s energy efficiency and speed in resource-constrained environments.

### Scalability and Adaptability  
- Design the system to scale across diverse datasets and adapt to industries like healthcare, retail, and entertainment.  

---

## Technology Stack  
1. **Programming Language**: Python  
2. **Libraries/Frameworks**: OpenCV, Keras, TensorFlow  
3. **Datasets**: UTKFace  
4. **Tools**: Akida Models, CNN2SNN, Matplotlib, NumPy  

---

## Expected Outcomes  

### Accurate and Robust Age Estimation  
- High-accuracy age prediction model validated using MAE.  
- Robust performance across diverse demographic groups.  

### Efficient Neuromorphic Inference  
- Functional age estimation model optimized for Akida neuromorphic hardware.  
- Demonstration of low power consumption and low latency.

### Real-Time Edge Deployment  
- Deployed system capable of real-time predictions on edge devices.  

### Improved User Experience  
- Enhanced personalization through an age-based recommendation system.  

---

## Introduction  
Neuromorphic computing, exemplified by BrainChip's Akida platform, mimics the brain's energy-efficient, event-driven processing. This project evaluates Akida’s potential for age estimation using the UTKFace dataset.  

### Key Steps:  
1. Train a VGG-based model using TensorFlow/Keras.  
2. Quantize the model for edge deployment.  
3. Convert it into Akida-compatible format.  
4. Compare Keras and Akida models on MAE and efficiency.  

---

## Background  
Age prediction from facial images is complex due to demographic variability. The UTKFace dataset supports model development, and Akida’s spike-based processing offers ultra-low-power inference with competitive accuracy. This project compares Keras and Akida models, emphasizing trade-offs between accuracy and efficiency.

---

## How to Test the Code and Set Up the Akida Development Environment  

### MetaTF Packages  
1. **Model Zoo (`akida-models`)**  
2. **Quantization Tool (`quantizeml`)**  
3. **Conversion Tool (`cnn2snn`)**  
4. **Akida Neuromorphic Processor Interface (`akida`)**  

### Installation  

1. Create and activate a virtual environment using Conda:  
   ```bash
   conda create --name akida_env python=3.11  
   conda activate akida_env  

    Install MetaTF packages:

pip install akida==2.11.0  
pip install cnn2snn==2.11.0  
pip install akida-models==1.6.2  

Alternatively, install dependencies from requirements.txt:

    pip install -r requirements.txt  

Key Steps in the Code

    Dataset Loading: Load UTKFace dataset and preprocess images.
    Model Evaluation: Test baseline, quantized, and Akida-converted models.
    Age Prediction: Predict age from custom images using all models.
    Optimization: Apply learning rate schedules and data augmentation.
    Edge Deployment: Optimize for edge devices with quantization and pruning.

Methodology

    Video Capture: Real-time capture using OpenCV.
    Image Processing: Preprocess and load data for training.
    Model Training: Train and optimize Keras models with quantization.
    Akida Conversion: Convert models for neuromorphic deployment.
    Testing: Evaluate on test images and custom inputs.

Contributions

    Implemented learning rate schedules for training optimization.
    Applied grid search for hyperparameter tuning.
    Incorporated data augmentation to improve robustness.
    Quantized models for edge deployment.
    Applied pruning for efficient edge device deployment.

Benefits

    Energy Efficiency: Neuromorphic inference with low power consumption.
    Scalability: Adaptable across datasets and industries.
    User Experience: Enhanced personalization through robust recommendations.
    
Conclusion

This project successfully demonstrated the development of an age-based recommendation system by leveraging the BrainChip Akida neuromorphic processor. Through a structured approach combining deep learning, quantization, neuromorphic computing, and efficient deployment techniques, we achieved several key objectives:

    Accurate Age Prediction:
    By training and optimizing a VGG-based model on the UTKFace dataset, we developed a reliable age estimation system with high accuracy and low Mean Absolute Error (MAE). The use of data augmentation, hyperparameter optimization, and quantization further enhanced robustness and generalization across diverse demographics.

    Energy-Efficient Inference:
    The conversion of the quantized Keras model to the Akida format demonstrated the power and efficiency of neuromorphic computing. Akida's event-driven architecture achieved real-time, low-power inference while maintaining comparable accuracy to traditional AI models.

    Seamless Model Conversion:
    The integration of the CNN-to-SNN framework allowed for smooth conversion of traditional models to Akida-compatible spiking models, showcasing the practicality of deploying advanced neuromorphic solutions on edge devices.

    Real-Time Edge Deployment:
    By implementing the system on an edge device, we demonstrated its ability to perform real-time age predictions and generate personalized recommendations dynamically. This highlights the feasibility of deploying AI solutions in resource-constrained environments like IoT devices and embedded systems.

    Scalability and Versatility:
    The modular design and scalability of the system make it adaptable for various industries, including healthcare, retail, and entertainment. Future extensions could integrate additional features like emotion recognition or gender-based personalization for enhanced user experiences.
