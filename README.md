# Image Captioning with Deep Learning

## Project Overview
This project implements an image captioning system using deep learning techniques. The model combines Computer Vision (CNN) and Natural Language Processing (LSTM) to generate descriptive captions for images automatically.

## Architecture
The system uses an **encoder-decoder architecture**:

### Encoder (Visual Feature Extraction)
- **VGG16** pre-trained on ImageNet
- Removes the top classification layers
- Uses Global Average Pooling to extract 512-dimensional feature vectors
- Acts as the "eyes" of the system

### Decoder (Language Generation)
- **LSTM (Long Short-Term Memory)** network
- 256 LSTM units for sequence generation
- Embedding layer with 128 dimensions
- Takes image features as initial state
- Generates captions word by word

### Model Flow
Image Input → VGG16 Feature Extraction → Dense(256) → LSTM Initial State
↓
Text Sequence → Embedding(128) → LSTM(256) → Dense(vocab_size) → Word Prediction

## How the Sequence Model Works

### Training Phase
1. **Image Features**: VGG16 extracts 512-dim feature vectors from input images
2. **Text Processing**: Captions are tokenized, padded, and converted to sequences
3. **Teacher Forcing**: During training, the model receives previous ground-truth words to predict next words
4. **Loss Calculation**: Uses categorical crossentropy to minimize prediction error

### Inference Phase
1. **Start Token**: Generation begins with "startseq" token
2. **Iterative Prediction**: At each step:
   - Current sequence is fed to LSTM
   - Model predicts probability distribution over vocabulary
   - Highest probability word is selected and appended to sequence
3. **Stop Condition**: Generation stops when "endseq" token is predicted or maximum length reached

### Key Sequence Concepts
- **Hidden State**: LSTM maintains context through hidden states initialized with image features
- **Word Embeddings**: Words are converted to dense vectors capturing semantic meaning
- **Sequence Length**: Maximum caption length of 25 words (determined from dataset)

## Implementation Details

### Data Pipeline
```python
# Image Processing
- Resize to 224x224 pixels
- VGG16 preprocessing (channel-wise centering)
- Feature extraction with frozen VGG16 weights

# Text Processing
- Convert to lowercase, remove punctuation
- Add special tokens: 'startseq' and 'endseq'
- Tokenization and vocabulary building
- Sequence padding to fixed length
```

## How to Run the code?

### Download Dependencies
` pip install tensorflow matplotlib numpy pillow `

### Data setup
1. Download flickr8k from https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
2. Update script path

`
IMAGE_DIR = "flickr8k/Images"\n
CAPTION_FILE = "flickr8k/captions.txt"
`

### Execute
`python image_captioning.py`





