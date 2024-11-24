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

Neuromorphic processors have developed from traditional computing systems, which relied on using binary logic and central processors to handle tasks within a step by step approach. While early computers were effective for routine things, mainly calculations, they struggled with complex tasks like pattern recognition and decision-making, which led researchers to explore brain-inspired models in the 1980s. Neuromorphic computing mimics the brain’s neural structure by using spiking neural networks (SNNs), that imitate the communication between neurons in the brain, allowing these processors to perform tasks such as pattern recognition, speech processing, and in this case, image analysis. As Artificial Intelligence (AI) and machine learning progressed, the need for systems capable of learning and adapting in real time propelled the development of neuromorphic processors. These chips are highly energy-efficient because they only consume power when processing relevant information, offering significant advantages in power efficiency and processing complex tasks
The Akida Brain chip developed by BrainChip represents a groundbreaking advancement in the field of neuromorphic computing, the level in which they engineered to replicate the brain's architecture and functionality in information processing has them arguably as the best in the field. This brain-like processing capability enables Akida to learn, adapt, and make independent decisions without the need for continuous retraining. The Akida architecture positions itself at the leading edge of AI innovation, expanding the possibilities of machine capabilities while ensuring practicality for widespread application across diverse industries. A significant technical feature of the Akida chip is its low power consumption, which is crucial for implementing AI in edge devices such as autonomous vehicles, drones, and smart sensors. This efficiency is realized through event-driven processing, allowing the chip to utilize power only during data processing, unlike traditional systems that consume energy continuously.  
In contrast to conventional processors that operate on a sequential basis, the Akida chip employs a parallel architecture capable of managing intricate, real-time data with remarkable efficiency. 
The Akida chip has the potential to revolutionize various industries by facilitating faster and more responsive AI applications, particularly in healthcare, where it can assist in real-time diagnostics and personalized treatment strategies. In the realm of robotics, Akida's capability to process sensory information and make instantaneous decisions could lead to the creation of more autonomous and adaptable robots. Akida enhances AI's ability to learn and function in dynamic environments, addressing many current challenges such as high computational costs and inefficiencies in learning from limited datasets. By providing a scalable and energy-efficient solution, Akida paves the way for the advancement of AI applications that demand both speed and sustainability. 
For this project we will be focusing on expanding the realm of the visual applications of Akida brain chip. Throughout the years, the ability to process visual data efficiently and in real-time has openings of a wide range of applications, from healthcare and security to agriculture and smart cities. Its neuromorphic design, which mimics brain-like processing, ensures that it can handle complex visual tasks more efficiently and intelligently than traditional computing systems. 

Background and Motivation

Introduced in 2017, Akida uses an event-driven, spiking neural network (SNN) architecture that mimics human cognition, allowing it to process facial images in real-time they represent a revolutionary leap in facial recognition technology, leveraging neuromorphic computing to replicate the brain's ability to process visual data efficiently. Unlike traditional image processing systems that rely on large datasets and constant data streaming, Akida can analyze facial features such as the eyes, nose, and mouth by processing sparse, event-driven inputs.. Akida’s ability to recognize faces under varying conditions—such as different lighting, angles, and facial expressions—sets it apart from conventional systems, which are typically less flexible and resource-efficient.
Facial recognition, a field that has evolved significantly over the past two decades, provides the context in which Akida’s capabilities shine. Initially, facial recognition systems relied on manual feature extraction methods in the 1990s, followed by more sophisticated techniques like eigenfaces and support vector machines in the 2000s. The major breakthrough came in the mid-2010s with the advent of deep learning algorithms and convolutional neural networks (CNNs), which greatly improved the accuracy and scalability of facial recognition systems. Akida represents the next step in this evolution by applying neuromorphic principles to achieve faster, more energy-efficient face identification. This allows Akida to outperform traditional AI models by processing data in real-time with minimal computational resources, which is crucial for applications like security systems, mobile devices, and personalized customer service.
The application of Akida in facial recognition extends across various industries, including security, healthcare, retail, and entertainment. In security systems, Akida’s real-time processing abilities can streamline access control and identification by accurately recognizing authorized personnel. 
This project involves using facial recognition to analyze a range of ages and generalize content for users on a social platform, it aims to broaden the appeal of content and ensure it is delivered ethically. By categorizing users based on age we seek to create a more inclusive content feed that appeals to a wider audience. In social platforms where personalized content often leads to echo chambers—where users are repeatedly exposed to content based on past behavior and interests, this approach introduces users to new topics and trends they might not encounter otherwise. This could foster discovery, encouraging users to explore diverse ideas and perspectives beyond their usual content preferences. By doing so, the platform would feel more inclusive and diverse, helping to attract a larger user base and improve overall engagement.
In addition to broadening content appeal, another key motivation for this project is ethical content delivery. By identifying the age group of each user, the platform can ensure that the content they see is age-appropriate and relevant. This helps protect younger users from exposure to material that may not be suitable for them, while ensuring older users are not overwhelmed with content that may be too juvenile or irrelevant. This approach can enhance the user experience by delivering content that resonates more directly with the user's life stage and interests, making their social platform experience feel more personal and engaging. Furthermore, this system could reduce the risk of harmful material being exposed to vulnerable groups, such as minors. Ultimately, the project is aimed at creating a safer, more engaging, and inclusive digital environment, where content is both varied and appropriate for a wide range of users.

