
# MULTImodal Misinformation & Disinformation Detection and Verification

## Overview

**MULTImodal Misinformation & Disinformation Detection and Verification** is a comprehensive project aimed at detecting and verifying fake news using multiple data sources, including text, images, audio, and video. This initiative leverages advanced machine learning techniques to identify and combat misinformation, particularly in the complex and dynamic context of the ongoing Ukrainian-Russian war. The project emphasizes the importance of reliable information in today's digital age, where misinformation can have serious consequences.

## Technologies Used

- **PyTorch**: For building and training neural networks, enabling efficient computation on both CPU and GPU.
- **Transformers**: Utilizing the DeBERTaV3 model for text processing, enhancing understanding of context and semantics in language.
- **OpenCV**: For image and video processing, facilitating tasks such as frame extraction and manipulation.
- **PyTesseract**: For optical character recognition (OCR), allowing the extraction of text from images.
- **TorchVision**: For image transformation and utilizing pre-trained models to improve feature extraction.
- **Pandas**: For data manipulation and analysis, essential for preprocessing and handling large datasets.
- **Scikit-learn**: For performance evaluation metrics, ensuring rigorous assessment of model accuracy and effectiveness.

## Dataset

The dataset utilized for this project is **MULTI-Fake-DetectiVE**, specifically curated to detect fake news and misleading information related to the Ukrainian-Russian war, which began in February 2022. This dataset comprises a diverse collection of real-world media content, enabling researchers and developers to advance the field of fake news detection in complex social and geopolitical contexts. It includes a range of formats, such as tweets, images, videos, and associated metadata, to provide a robust training foundation for multimodal analysis.

The dataset's four classes—Fake, Real, Probably Fake, and Probably Real—better reflect real-world misinformation challenges than a simple Real/Fake split. This classification helps the model understand varying levels of credibility, making it more effective at identifying not just clear falsehoods but also content with ambiguity. This approach supports more accurate and practical misinformation detection for real-world combat related applications.

[Data Source](https://sites.google.com/unipi.it/multi-fake-detective/data?authuser=0)


## Project Structure
The Structured branch hosts the **Streamlit** application, organized in the following file structure:

```plaintext
baseline_models/
├── Test_video.mp4                  # Sample test video for baseline model testing
├── people.mp4                      # Additional video file for testing or model input
├── face_extract.py                 # Script for extracting faces from video content
├── baseline1/                      # Directory for the first baseline model
│   └── Baseline_Code1.ipynb        # Notebook implementing the first baseline model
└──baseline2/                      # Directory for the second baseline model
    └── BaseLine2.ipynb             # Notebook implementing the second baseline model

config/
└── settings.py                     # Python-based settings for easy access to configuration variables

multimodal-fake-news-detection/
├── backend/
│   ├── multimodal_model.py         # Model architecture and training scripts for multimodal processing
│   ├── utils.py                    # Utility functions for data preprocessing and model evaluation
│   ├── requirements.txt            # Dependencies for backend
│   ├── Video_to_text.py            # Script for converting video content to text
│   ├── videos_images_to_text.py    # Extracts text from both video frames and images
│   ├── deepfake_model.py           # Model or methods for detecting deepfakes
│   ├── face_extraction.py          # Extracts faces from video/image data
│   ├── text_extraction.py          # Handles text extraction processes
│   ├── train.py                    # Script for training the multimodal model
├── frontend/
│   ├── app.py                      # Streamlit application for user interaction
└── main.py                         # Entry point for running the entire Streamlit application

README.md                       # Project overview, setup instructions, and documentation

## How the Baseline_Code1 Works 

1. **Frontend**: Users provide inputs through a user-friendly Streamlit interface, uploading media content (text, images, audio, or video) for analysis.
2. **Backend Model**: The provided inputs are processed by our trained models:
   - **Text Encoder**: Utilizes the DeBERTaV3 model to process textual data, capturing nuances in language and sentiment.
   - **Image Encoder**: Implements DenseNet-121 for robust feature extraction from images, identifying relevant visual cues.
   - **Multimodal Fusion**: Combines features from both text and images using a Transformer model, enhancing the model's ability to understand context.
   - **Deepfake Detection**: An additional module that employs advanced techniques to identify manipulated video content.
   Additionally, we have functionalities to extract text and images from videos and identify their Realness by passing them to the above models/Encoders. This is a future scope that can be worked on further.
3. **Output Rendering**: The results are returned to the main page, providing users with insights on the likelihood of misinformation, contextual explanations, and confidence scores.

## Training

To train the model, execute the following commands in your terminal:

```bash
pip install -r backend/requirements.txt
python backend/train.py
```

- Adjust parameters in `train.py` to modify training epochs, batch size, and learning rates based on your system’s capabilities.

## Evaluation

The model's performance is evaluated using various metrics, including accuracy, F1 score, precision, and recall. Detailed training logs and evaluation reports will be generated during the training process. Comprehensive classification reports are also available to assess the effectiveness of the model across different classes.

The Baseline_Code1 Model has reached a high accuracy of 70.77% which is the best compared to models released across the internet trained on the same dataset. The below link compares the best models trained on this dataset.
[Data Source](https://ceur-ws.org/Vol-3473/paper33.pdf)

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or wish to report issues, please open an issue or submit a pull request. We encourage collaborative efforts to enhance the project's capabilities and reach.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Future Work

- **Integration of Additional Modalities**: Expand capabilities to include audio and video analysis more comprehensively.
- **Real-time Detection**: Implement mechanisms for real-time misinformation detection using live feeds from social media platforms.
- **User Feedback Mechanism**: Incorporate a user feedback system to continuously improve model accuracy based on community input.
