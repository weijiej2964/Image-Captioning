# image_captioning.py
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PIL import Image
import glob
import matplotlib.pyplot as plt


class ImageCaptioningModel:
    def __init__(self):
        self.tokenizer = None
        self.max_length = None
        self.vocab_size = None
        self.model = None
        self.image_to_captions = {}  # Store captions for lookup

    def display_image_with_captions(self, image_path, generated_caption, actual_captions):
        """Display the image with generated and actual captions"""
        try:
            img = Image.open(image_path)

            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Generated Caption: '{generated_caption}'",
                      fontsize=14, color='red', pad=20)
            plt.axis('off')

            # Create caption text
            if actual_captions:
                caption_text = "Actual Captions:\n" + \
                    "\n".join([f"• {cap}" for cap in actual_captions])
            else:
                caption_text = "No actual captions found for this image"

            plt.figtext(0.5, 0.05, caption_text, ha="center", fontsize=12,
                        bbox={"facecolor": "lightblue", "alpha": 0.7, "pad": 5})

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Could not display image: {e}")

    def load_flickr8k_data(self, image_dir, caption_file, num_samples=100):
        """Load Flickr8k dataset - FIXED for your specific structure"""
        print("Loading Flickr8k data...")

        # Load and parse captions
        self.image_to_captions = {}  # Reset and store for later use
        with open(caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Skip header and parse each line
        for line in lines[1:]:  # Skip the first line (header: "image,caption")
            line = line.strip()
            if not line:
                continue

            # Split by comma
            parts = line.split(',', 1)  # Split only on first comma
            if len(parts) == 2:
                image_name, caption = parts
                image_name = image_name.strip()
                caption = caption.strip()

                if image_name not in self.image_to_captions:
                    self.image_to_captions[image_name] = []
                self.image_to_captions[image_name].append(caption)

        print(f"Found {len(self.image_to_captions)} images in captions file")

        # Match with actual image files
        images = []
        all_captions = []

        # Get all image files
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        print(f"Found {len(image_files)} image files")

        # Match images with captions
        matched_count = 0
        for img_file in image_files[:num_samples]:
            img_name = os.path.basename(img_file)

            if img_name in self.image_to_captions:
                images.append(img_file)
                for caption in self.image_to_captions[img_name]:
                    all_captions.append(caption)
                matched_count += 1

        print(f"Matched {matched_count} images with captions")
        print(f"Total: {len(images)} images, {len(all_captions)} captions")
        return images, all_captions

    def get_actual_captions_for_image(self, image_name):
        """Get actual captions for a specific image"""
        if image_name in self.image_to_captions:
            # Return the original captions (without our added tokens)
            return self.image_to_captions[image_name]
        return []

    def preprocess_captions(self, captions):
        """Preprocess captions and create tokenizer"""
        print("Preprocessing captions...")

        # Clean captions
        cleaned_captions = []
        for caption in captions:
            caption = caption.lower().strip()
            # Remove any trailing period and add sequence tokens
            caption = caption.rstrip('.')
            caption = 'startseq ' + caption + ' endseq'
            cleaned_captions.append(caption)

        # Create tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(cleaned_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Find maximum sequence length
        sequences = self.tokenizer.texts_to_sequences(cleaned_captions)
        self.max_length = max(len(seq) for seq in sequences)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Max sequence length: {self.max_length}")

        return cleaned_captions

    def extract_image_features(self, image_paths):
        """Extract features using VGG16"""
        print("Extracting image features...")

        # Load pre-trained VGG16 with average pooling
        base_model = VGG16(weights='imagenet',
                           include_top=False, pooling='avg')

        features = {}
        successful = 0

        for img_path in image_paths:
            try:
                # Load and preprocess image
                img = Image.open(img_path)
                img = img.resize((224, 224))

                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.vgg16.preprocess_input(
                    img_array)

                # Extract features
                feature = base_model.predict(img_array, verbose=0)

                img_name = os.path.basename(img_path)
                features[img_name] = feature[0]
                successful += 1

                if successful % 10 == 0:
                    print(
                        f"  Processed {successful}/{len(image_paths)} images...")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        print(
            f"Successfully extracted features for {successful}/{len(image_paths)} images")
        return features

    def create_training_data(self, image_paths, captions, image_features):
        """Create training data"""
        print("Creating training data...")

        X1, X2, y = [], [], []

        # Group captions by image
        image_captions_dict = {}
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            # Get all captions for this image (5 per image in Flickr8k)
            start_idx = i * 5
            end_idx = start_idx + 5
            image_captions_dict[img_name] = captions[start_idx:end_idx]

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            if img_name not in image_features:
                continue

            img_feature = image_features[img_name]

            # Use up to 3 captions per image for training
            for caption in image_captions_dict.get(img_name, [])[:3]:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                # Create input-output pairs
                for k in range(1, len(seq)):
                    in_seq = seq[:k]
                    out_word = seq[k]

                    # Pad input sequence
                    in_seq_padded = pad_sequences(
                        [in_seq], maxlen=self.max_length, padding='pre')[0]

                    # Convert output to one-hot
                    out_onehot = to_categorical(
                        [out_word], num_classes=self.vocab_size)[0]

                    X1.append(img_feature)
                    X2.append(in_seq_padded)
                    y.append(out_onehot)

        print(f"Created {len(X1)} training samples")
        return np.array(X1), np.array(X2), np.array(y)

    def build_model(self, embedding_dim=128, lstm_units=256):
        """Build the image captioning model - FIXED dimension mismatch"""
        print("Building model...")

        # Image feature input (512 with average pooling)
        image_input = Input(shape=(512,))
        # FIX: Make image dense output match LSTM units for initial state
        image_dense = Dense(lstm_units, activation='relu')(image_input)

        # Sequence input
        caption_input = Input(shape=(self.max_length,))
        caption_embedding = Embedding(
            self.vocab_size, embedding_dim)(caption_input)

        # LSTM decoder - FIX: initial state now matches LSTM units
        lstm_output = LSTM(lstm_units)(caption_embedding,
                                       initial_state=[image_dense, image_dense])

        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(lstm_output)

        # Create model
        self.model = Model(inputs=[image_input, caption_input], outputs=output)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        print("Model built successfully!")
        print(f"Total parameters: {self.model.count_params()}")

    def train(self, X1, X2, y, epochs=5, batch_size=32, validation_split=0.1):
        """Train the model"""
        if len(X1) == 0:
            print("No training data available!")
            return None

        print(f"Starting training with {len(X1)} samples...")

        history = self.model.fit(
            [X1, X2], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        return history

    def generate_caption(self, image_path, image_features):
        """Generate caption for a new image"""
        if self.model is None:
            return "Model not trained yet"

        try:
            img_name = os.path.basename(image_path)
            if img_name not in image_features:
                return "Image features not found"

            img_feature = image_features[img_name]

            # Start with start token
            in_text = 'startseq'

            for i in range(self.max_length):
                # Tokenize input sequence
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences(
                    [sequence], maxlen=self.max_length, padding='pre')

                # Predict next word
                yhat = self.model.predict(
                    [np.array([img_feature]), sequence], verbose=0)
                yhat = np.argmax(yhat)

                # Convert index to word
                word = self.tokenizer.index_word.get(yhat, None)

                # Stop if we cannot map the word or reach end token
                if word is None or word == 'endseq':
                    break

                in_text += ' ' + word

            # Clean up the caption
            caption = in_text.replace('startseq ', '').replace(' endseq', '')
            return caption

        except Exception as e:
            return f"Error: {e}"


def main():
    # Initialize model
    caption_model = ImageCaptioningModel()

    # Set paths based on your structure
    IMAGE_DIR = "flickr8k/Images"
    CAPTION_FILE = "flickr8k/captions.txt"

    print("=== Flickr8k Image Captioning ===")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Caption file: {CAPTION_FILE}")

    images, captions = caption_model.load_flickr8k_data(
        IMAGE_DIR, CAPTION_FILE, num_samples=200)

    if len(images) == 0:
        print("ERROR: No images loaded! Check file paths and structure.")
        return

    print(
        f"\nSuccessfully loaded {len(images)} images with {len(captions)} captions")

    # Preprocess captions
    cleaned_captions = caption_model.preprocess_captions(captions)

    # Extract image features
    image_features = caption_model.extract_image_features(images)

    # Create training data
    X1, X2, y = caption_model.create_training_data(
        images, cleaned_captions, image_features)

    if len(X1) == 0:
        print("ERROR: No training data created!")
        return

    print(f"\nTraining data ready: {len(X1)} samples")

    # Build model
    caption_model.build_model(embedding_dim=128, lstm_units=256)

    # Train model
    print("\nStarting training...")
    history = caption_model.train(X1, X2, y, epochs=15, batch_size=16)

    # Test caption generation
    if len(images) > 0:
        test_image = images[1]
        test_image_name = os.path.basename(test_image)
        print(f"\n=== Testing Caption Generation ===")
        print(f"Test image: {test_image_name}")

        # Generate caption
        generated_caption = caption_model.generate_caption(
            test_image, image_features)
        print(f"Generated caption: {generated_caption}")

        # Get actual captions using the stored mapping
        actual_captions = caption_model.get_actual_captions_for_image(
            test_image_name)

        print(f"\nActual captions for {test_image_name}:")
        if actual_captions:
            for i, caption in enumerate(actual_captions, 1):
                print(f"  {i}. {caption}")
        else:
            print("  No actual captions found!")

        # Display the image with captions
        print(f"\nDisplaying image with captions...")
        caption_model.display_image_with_captions(
            test_image, generated_caption, actual_captions)

    print("\n=== Training Complete ===")
    print("You can now:")
    print("1. Increase num_samples for better results")
    print("2. Train for more epochs")
    print("3. Use the generate_caption() method on new images")


if __name__ == "__main__":
    main()
