
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

## Project Structure

```
multimodal-fake-news-detection/
├── backend/
│   ├── model.py              # Model architecture and training scripts
│   ├── utils.py              # Utility functions for data preprocessing and model evaluation
│   ├── requirements.txt      # Dependencies for backend
│   └── train.py              # Script to train the model
├── frontend/
│   ├── app.py                # Streamlit application for user interaction
│   ├── static/               # Static files (images, CSS, etc.)
│   ├── templates/            # HTML templates for rendering output
└── README.md                 # Project documentation
```

- **backend/**: Contains the model architecture, utility functions, training scripts, and dependencies required to run the model.
- **frontend/**: Contains the Streamlit application that allows users to interact with the model and view results.

## How It Works

1. **Frontend**: Users provide inputs through a user-friendly Streamlit interface, uploading media content (text, images, audio, or video) for analysis.
2. **Backend Model**: The provided inputs are processed by our trained models:
   - **Text Encoder**: Utilizes the DeBERTaV3 model to process textual data, capturing nuances in language and sentiment.
   - **Image Encoder**: Implements DenseNet-121 for robust feature extraction from images, identifying relevant visual cues.
   - **Multimodal Fusion**: Combines features from both text and images using a Transformer model, enhancing the model's ability to understand context.
   - **Deepfake Detection**: An additional module that employs advanced techniques to identify manipulated video content.
3. **Output Rendering**: The results are rendered back to the main page, providing users with insights on the likelihood of misinformation, along with contextual explanations and confidence scores.

## Training

To train the model, execute the following commands in your terminal:

```bash
pip install -r backend/requirements.txt
python backend/train.py
```

- Adjust parameters in `train.py` to modify training epochs, batch size, and learning rates based on your system’s capabilities.

## Evaluation

The model's performance is evaluated using various metrics, including accuracy, F1 score, precision, and recall. Detailed training logs and evaluation reports will be generated during the training process and can be found in the output directory specified in `train.py`. Comprehensive classification reports are also available to assess the effectiveness of the model across different classes.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or wish to report issues, please open an issue or submit a pull request. We encourage collaborative efforts to enhance the project's capabilities and reach.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Future Work

- **Integration of Additional Modalities**: Expand capabilities to include audio and video analysis more comprehensively.
- **Real-time Detection**: Implement mechanisms for real-time misinformation detection using live feeds from social media platforms.
- **User Feedback Mechanism**: Incorporate a user feedback system to continuously improve model accuracy based on community input.
