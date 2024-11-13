# Image Prediction with TensorFlow

This project uses a pre-trained model to predict the category of an input image. Follow the instructions below to set up your environment and run the prediction script.

## Prerequisites

Ensure you have the following installed on your machine:

- Python 3.9 or higher
- Conda package manager

## Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Create a Conda environment with a specific name (e.g., `image_predict_env`) and install the required dependencies:
   ```bash
   conda create -n image_predict_env python=3.9 -f environment.yml
   ```

3. Verify the setup by running:
   ```bash
   python --version
   ```
   Ensure Python 3.9 or higher is being used within the Conda environment.

## Usage

Run the following command to predict the category of an image:

```bash
python predict.py ./test_images/orange_dahlia.jpg trained_model --category_names label_map.json --top_k 2
```

### Command Breakdown:

- `./test_images/orange_dahlia.jpg`: Path to the image you want to classify.
- `trained_model`: The pre-trained model file (e.g., `trained_model.h5`).
- `--category_names label_map.json`: JSON file mapping category labels to their names.
- `--top_k 2`: Number of top predictions to display.

### Example Output:

```
english marigold: 0.3580
orange dahlia: 0.3448
mexican aster: 0.0461
osteospermum: 0.0412
gazania: 0.0404
```

## Notes

- Ensure all required files (e.g., `trained_model.h5`, `label_map.json`, and the `test_images` directory) are present in the project directory.
- If the `predict.py` script encounters issues, ensure the Python environment and dependencies are correctly set up.
