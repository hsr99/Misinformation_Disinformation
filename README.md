
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

## Model Documentation
Click the below link to view our model documentation

[Documentation of the model](https://docs.google.com/document/d/13h27LtKl0stQLokG4fonuRGf5_FaCZkqmhU0ubFqnp8/edit?usp=sharing)

## Project Structure
The main branch hosts the **Streamlit** application, organized in the following file structure:

```plaintext
multimodal-fake-news-detection/
├── backend/
│   ├── model.py              # Model architecture and training scripts
│   ├── utils.py              # Utility functions for data preprocessing and model evaluation
│   ├── requirements.txt      # Dependencies for backend
│   └── train.py              # Script to train the model
├── frontend/
│   ├── app.py                # Streamlit application for user interaction
│   ├── static/               # Static files (images, CSS, etc.)
│   └── templates/            # HTML templates for rendering output
└── README.md                 # Project documentation
```

## Baseline Model

**Baseline_Code1** offers the best performance:
- **Text Encoder**: Processes textual data with the DeBERTaV3 model.
- **Image Encoder**: Employs DenseNet-121 for feature extraction from images.
- **Multimodal Fusion**: Integrates features from text and images via a Transformer model.
- **Deepfake Detection**: Detects manipulated video content using advanced techniques.

The Baseline_Code1 model has achieved a high accuracy of 70.77%—the highest compared to other models on the same dataset. More details on model comparison are available in the following [paper](https://ceur-ws.org/Vol-3473/paper33.pdf).

## Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
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
