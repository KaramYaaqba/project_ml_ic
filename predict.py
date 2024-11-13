import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

def load_model(model_path):
    model_name = f"{model_path}.h5"
    loaded_model = tf.keras.models.load_model(model_name, custom_objects={'KerasLayer': hub.KerasLayer})
    loaded_model.summary()
    return loaded_model

def process_image(image):
    """
    Processes an image to be compatible with the model's input requirements.

    Args:
    image (np.array): Input image in the form of a NumPy array.

    Returns:
    np.array: Preprocessed image as a NumPy array with shape (224, 224, 3).
    """
    # Convert the image to a Tensor
    image = tf.convert_to_tensor(image)

    # Resize to (224, 224)
    image = tf.image.resize(image, (224, 224))

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    # Convert back to a NumPy array
    return image.numpy()

def predict(image_path, model, top_k):
    # Load and preprocess the image
    image = Image.open(image_path)

    # Convert to NumPy array
    image = np.asarray(image)

    # Preprocess the image
    processed_image = process_image(image)

    # Add a batch dimension
    processed_image = np.expand_dims(processed_image, axis=0)

    # Make predictions
    predictions = model.predict(processed_image)
    predictions = predictions[0]

    # Get the top 5 predictions
    top_5_indices = np.argsort(predictions)[-5:][::-1]  # Indices of top 5 classes
    top_5_probs = predictions[top_5_indices]

    return top_5_probs, top_5_indices

def map_labels(labels, category_names):
    with open(category_names, 'r') as f:
        label_map = json.load(f)
    return [label_map[str(label)] for label in labels]

def main():
    parser = argparse.ArgumentParser(description='Predict the top flower classes for an input image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('model', type=str, help='Path to the trained model file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model)

    # Predict the top K classes
    probabilities, classes = predict(args.image_path, model, args.top_k)

    # Map the labels if category names are provided
    if args.category_names:
        class_names = map_labels(classes, args.category_names)
    else:
        class_names = classes

    # Display the results
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {probabilities[i]:.4f}")

if __name__ == '__main__':
    main()
