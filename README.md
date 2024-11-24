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
(Provide a short description of the problem you're addressing)

Spiking neural networks (SNNs) are expected to
be part of the future AI portfolio, with heavy investment from
industry and government, e.g. Loihi.While
Artificial Neural Network (ANN) architectures have taken large
strides, few works have targeted SNN hardware efficiency. Our
analysis of SNN baselines shows that at modest spike rates,
SNN implementations exhibit significantly lower efficiency than
accelerators for ANNs. 

## Key Objectives:
•	 Clarify our main goals for using neuromorphic computing Literature Review: Research similar work in neuromorphic architecture, and Spiking Neural Networks (SNNs) to understand current approaches, datasets, and tools. 
•	Select Tools and Libraries: Decide on simulation software (e.g., BindsNET, NEST, or SpiNNaker simulator) and any supporting libraries (e.g., PyTorch) based on compatibility and ease of use.
•	SNN Model Selection: Choose a suitable SNN architecture (e.g., convolutional SNN, recurrent SNN) for gesture recognition.
•	Model Training: Use a framework like BindsNET, which allows for integrating SNN models with conventional neural network training methods.
•	Optimization: Experiment with different SNN parameters (e.g., neuron types, spike thresholds) to optimize for real-time performance.

## Technology Stack:
(List the hardware platform, software tools, language(s), etc. you plan to use)
- Google collab
- Python
- Jupyter Notebook

## Expected Outcomes:
Analyze how different spike-coding schemes, network parameters, and simulator settings affect performance.

## Timeline:
(Provide a rough timeline or milestones for the project)
- Week 1 (11/4/2024 - 11/10/2024): Proposal and Presentation
- Week 2 (11/11/2024 - 11/17/2024): Low-Bit Quantization Optimization Experiment
- Week 3 (11/18/2024 - 11/24/2024): Low-Bit Quantization Optimization Experiment
- Week 4 (11/25/2024 - 12/01/2024): Extending CNN Models on the Hive Platform
- Week 5 (12/02/2024 - 12/08/2024): Extending CNN Models on the Hive Platform

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

