Image Captioning with Deep Learning
Project Overview
This project implements an image captioning system using deep learning techniques. The model combines Computer Vision (CNN) and Natural Language Processing (LSTM) to generate descriptive captions for images automatically.

Architecture
The system uses an encoder-decoder architecture:

Encoder (Visual Feature Extraction)
VGG16 pre-trained on ImageNet

Removes the top classification layers

Uses Global Average Pooling to extract 512-dimensional feature vectors

Acts as the "eyes" of the system

Decoder (Language Generation)
LSTM (Long Short-Term Memory) network

256 LSTM units for sequence generation

Embedding layer with 128 dimensions

Takes image features as initial state

Generates captions word by word

Model Structure
text
Input Image → VGG16 Feature Extraction → Dense Layer → LSTM Initial State
                    ↓
Text Sequence → Embedding Layer → LSTM → Dense Output → Next Word Prediction
How It Works
Image Processing: Input images are resized to 224x224 and fed through VGG16

Feature Extraction: VGG16 extracts visual features (512-dim vectors)

Sequence Generation: LSTM uses image features as context to generate captions

Word Prediction: At each step, the model predicts the next word based on previous words and image context

Beam Search: Generates captions by sequentially predicting words until end token

Installation & Setup
Prerequisites
bash
pip install tensorflow matplotlib numpy pillow
Dataset Setup
Download Flickr8k dataset from Kaggle

Update file paths in image_captioning.py:

python
IMAGE_DIR = "path/to/flickr8k/Images"
CAPTION_FILE = "path/to/flickr8k/captions.txt"
Running the Model
bash
python image_captioning.py
Key Features
Transfer Learning: Uses pre-trained VGG16 for visual understanding

Sequence Modeling: LSTM handles variable-length caption generation

Teacher Forcing: Training technique for better convergence

Flexible Input: Works with any image size (automatically resized)

Visualization: Displays images with generated vs actual captions

Results & Performance
Sample Outputs
Input Image: Child climbing stairs

Generated Caption: "a child in a pink dress is climbing up stairs"

Actual Captions:

"A child in a pink dress is climbing up a set of stairs"

"A girl going into a wooden building"

"A little girl climbing into a wooden playhouse"

Model Metrics
Vocabulary Size: 500-1000 words (depending on training data)

Sequence Length: Up to 25 words

Training Time: 15-60 minutes (depending on dataset size)

Accuracy: Improves with more data and epochs

Technical Details
Data Preprocessing
Image normalization using VGG16 preprocessing

Text tokenization and sequence padding

Vocabulary building from captions

Train/validation split (90/10)

Training Parameters
Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 16-32

Epochs: 15-30 (recommended)

Embedding Dimension: 128

LSTM Units: 256

Model Configuration
python
{
    "embedding_dim": 128,
    "lstm_units": 256,
    "feature_size": 512,
    "max_sequence_length": 25,
    "vocab_size": 516  # varies with data
}
Team Contributions
Team Member 1: Data loading and preprocessing pipeline

Team Member 2: CNN feature extraction implementation

Team Member 3: LSTM architecture and training loop

Team Member 4: Caption generation and visualization

Challenges & Solutions
Challenges Faced
Dimension Mismatch: LSTM initial state size didn't match hidden units

Solution: Made dense layer output match LSTM units

Caption Matching: Difficulty linking images with correct captions

Solution: Implemented exact filename matching

Training Time: Slow convergence with small datasets

Solution: Increased dataset size and epochs

Lessons Learned
Pre-trained CNNs significantly boost performance

LSTM initial state is crucial for conditioning on images

More training data dramatically improves caption quality

Proper data preprocessing is essential for good results

Future Improvements
Use larger datasets (Flickr30k, COCO)

Implement attention mechanisms

Try transformer-based architectures

Add beam search for better caption generation

Fine-tune CNN layers for better feature extraction

Files Structure
text
project/
├── image_captioning.py      # Main implementation
├── requirements.txt         # Dependencies
├── README.md               # This file
└── flickr8k/               # Dataset
    ├── Images/
    └── captions.txt
References
Flickr8k Dataset

TensorFlow Image Captioning Tutorial

VGG16 Paper

LSTM Paper

License
This project is for educational purposes as part of CSCI 35000 - Deep Learning course.
