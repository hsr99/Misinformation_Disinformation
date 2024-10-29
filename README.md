

# MULTImodal Misinformation & Disinformation Detection and Verification

## Overview

MULTImodal Misinformation & Disinformation Detection and Verification is a comprehensive project aimed at detecting and verifying fake news using multiple data sources, including text, images, audio, and video. This project leverages advanced machine learning techniques to identify misinformation, especially in the context of the ongoing Ukrainian-Russian war.

## Technologies Used

- **PyTorch**: For building and training neural networks.
- **Transformers**: Utilizing the DeBERTaV3 model for text processing.
- **OpenCV**: For image and video processing.
- **PyTesseract**: For optical character recognition (OCR).
- **TorchVision**: For image transformation and pre-trained models.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For performance evaluation metrics.

## Dataset

The dataset used for this project is **MULTI-Fake-DetectiVE**, specifically curated to detect fake news and misleading information related to the Ukrainian-Russian war, which began in February 2022. It provides a diverse collection of real-world media content, enabling researchers and developers to advance the field of fake news detection in complex social and geopolitical contexts.

## Project Structure

```
multimodal-fake-news-detection/
├── backend/
│   ├── model.py
│   ├── utils.py
│   ├── requirements.txt
├── frontend/
│   ├── app.py
│   ├── static/
│   ├── templates/
└── README.md
```

- **backend/**: Contains the model architecture, utility functions, and dependencies.
- **frontend/**: Contains the Streamlit application for user interaction.

## How It Works

1. **Frontend**: Users provide inputs through the Streamlit interface.
2. **Backend Model**: The provided inputs are processed by our trained models:
   - **Text Encoder**: Utilizes DeBERTaV3 for processing textual data.
   - **Image Encoder**: Uses DenseNet-121 for image feature extraction.
   - **Multimodal Fusion**: Combines features from text and images using a Transformer model.
   - **Deepfake Detection**: An additional module for detecting deepfake content.
3. **Output Rendering**: The results are rendered back to the main page, providing users with insights on misinformation.

## Training

To train the model, execute the following commands in your terminal:

```bash
pip install -r backend/requirements.txt
python backend/model.py
```

## Evaluation

The model's performance is evaluated using accuracy, F1 score, and other relevant metrics. You can find detailed training logs and evaluation reports in the output generated during training.

## Contributing

Contributions are welcome! If you have suggestions for improvements or would like to report issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

