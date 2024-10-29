# Misinformation_Disinformation
MULTImodal Fake News Detection and Verification 

Dataset Description: MULTI-Fake-DetectiVE is a comprehensive dataset specifically curated to detect fake news and misleading information in the context of the Ukrainian-Russian war, which began in February 2022. Recognizing the increased dissemination of fake news since the early stages of this conflict, this dataset provides researchers and developers with a diverse collection of real-world media content, enabling them to advance the field of fake news detection in complex social and geopolitical contexts.

In this project, we tackle the challenge of identifying misinformation by utilizing textual and visual data from tweets. This solution is based on two main components:
  ->A text encoder using the DeBERTaV3 Large model.
  ->An image encoder using DenseNet-121 for robust feature extraction.
These two embeddings are then passed through a Transformer encoder for feature fusion and later classified into the target classes.

Install the dependencies using pip install -r requirements.txt

Usage
Set CUDA for Debugging: %env CUDA_LAUNCH_BLOCKING=1
Train the Model: Run the training loop using the main script.
