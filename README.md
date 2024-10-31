
# MULTImodal Misinformation & Disinformation Detection and Verification

## Overview

**MULTImodal Misinformation & Disinformation Detection and Verification** is an innovative project aimed at detecting and verifying fake news by analyzing multiple data sources—text, images, audio, and video. Designed with advanced machine learning techniques, this project targets misinformation in complex situations like the ongoing Ukrainian-Russian conflict, emphasizing the critical role of factual information in today’s digital world.

## Technologies Used

- **PyTorch**: Powers neural network creation and training, supporting both CPU and GPU-based computation.
- **Transformers (DeBERTaV3)**: Processes text data, capturing intricate nuances in language and sentiment.
- **OpenCV**: Facilitates image and video processing, including frame extraction and analysis.
- **PyTesseract**: Provides optical character recognition (OCR) for extracting text from images.
- **TorchVision**: Offers image transformation and feature extraction capabilities with pre-trained models.
- **Pandas**: Manages data manipulation, essential for preprocessing and handling large datasets.
- **Scikit-learn**: Provides metrics for evaluating model performance, ensuring accuracy and reliability.

## Dataset

The **MULTI-Fake-DetectiVE** dataset is specially curated to detect fake news and misleading information related to the Ukrainian-Russian conflict. It includes diverse media formats (e.g., tweets, images, videos) and metadata, enabling comprehensive multimodal analysis. Classes in the dataset—Fake, Real, Probably Fake, Probably Real—allow the model to assess credibility on a nuanced scale, making it adept at identifying both clear falsehoods and ambiguous information.

[Dataset Source](https://sites.google.com/unipi.it/multi-fake-detective/data?authuser=0)


## Project Structure
The main branch hosts the **Streamlit** application, organized in the following file structure:

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

```

## Baseline Model

**Baseline_Code1** offers the best performance:
- **Text Encoder**: Processes textual data with the DeBERTaV3 model.
- **Image Encoder**: Employs DenseNet-121 for feature extraction from images.
- **Multimodal Fusion**: Integrates features from text and images via a Transformer model.
- **Deepfake Detection**: Detects manipulated video content using advanced techniques.

The Baseline_Code1 model has achieved a high accuracy of 70.77%—the highest compared to other models on the same dataset. Page 5 of the following [paper](https://ceur-ws.org/Vol-3473/paper33.pdf) provides more details on model comparison.

## Running the Project

### 1. Clone the Repository

```bash
git clone <[repository-url](https://github.com/hsr99/Misinformation_Disinformation.git)>
cd multimodal-fake-news-detection
```

### 2. Install Dependencies

Navigate to the backend directory and install the required Python packages:

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

Launch the Streamlit application to access the prototype:

```bash
streamlit run frontend/app.py
```

The application will be accessible in your browser at `http://localhost:8501`.


```bash
python backend/train.py
```

## Evaluation

Model evaluation is performed using metrics such as accuracy, F1 score, precision, and recall. Logs and reports are generated during training to monitor the model’s effectiveness across different classes.

## Future Work

- **Expanded Modalities**: Enhance support for audio and video analysis.
- **Real-time Detection**: Implement real-time misinformation detection via live social media feeds.
- **User Feedback**: Add a feedback system to improve model accuracy through user contributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are encouraged! Open an issue or submit a pull request to propose new features, suggest improvements, or report issues.
