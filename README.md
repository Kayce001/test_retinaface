# RetinaFace Image Assessment Project

This project involves using RetinaFace, a state-of-the-art face detection model, to analyze images in a specified directory and report the number of faces detected in each. Optionally, the detected counts can be compared to a ground truth file to calculate the accuracy of the face detection.

## Dependencies

- Python 3.10+
- OpenCV
- PyTorch
- NumPy
- Ubuntu 20.04 or later

## Installation

There are two methods to set up the environment for this project:

### Method 1: Using Conda Environment File

1. Clone the Pytorch_Retinaface repository:
   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```
2. Change directory to the cloned repository:
   ```
   cd Pytorch_Retinaface
   ```
3. Place the `environment.yml`, `unittest_face_detection.py`, and `script.py` files into the `Pytorch_Retinaface` directory.
4. Create the conda environment and activate it:
   ```bash
   conda env create -f environment.yml
   conda activate retinaface
   ```
5. Follow the instructions in the Pytorch_Retinaface repository to download and set up the RetinaFace model in the `./weights` directory.

### Method 2: Manual Installation

1. Ensure Python 3.10 or later is installed and you are using Ubuntu 20.04 or newer.
2. Clone the Pytorch_Retinaface repository:
   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```
3. Change directory to the cloned repository:
   ```
   cd Pytorch_Retinaface
   ```
4. Place the `requirements.txt`, `unittest_face_detection.py`, and `script.py` files into the `Pytorch_Retinaface` directory.
5. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
6. Follow the instructions in the Pytorch_Retinaface repository to download and set up the RetinaFace model in the `./weights` directory.

## Usage

Execute the script with the following command format:

```
python script.py --input_folder /path/to/images --output_file /path/to/output_file [--truth /path/to/truth_file] [--ground_truth /path/to/ground_truth_folder] [--m /path/to/model] [--network <network>] [--confidence_threshold <confidence_threshold>] [--nms_threshold <nms_threshold>] [--cpu]
```

### Options:

- `--input_folder`: Directory containing the images.
- `--output_file`: File path for the results.
- `--truth`: (Optional) File with ground truth labels for face counts.
- `--ground_truth`: (Optional) Directory with ground truth bounding boxes.
- `--m`: Path to the trained model file (default: './weights/mobilenet0.25_Final.pth').
- `--network`: Backbone network ('mobile0.25' or 'resnet50', default: 'mobile0.25').
- `--confidence_threshold`: Confidence threshold for detection (default: 0.6).
- `--nms_threshold`: Non-maximum suppression threshold (default: 0.4).
- `--cpu`: Enable CPU inference (if no GPU is available).

The `--truth` file format:

```
image1.jpg 3
image2.jpg 1
image3.jpg 0
...
```

The `--ground_truth` directory should contain `.txt` files named after the image files, with space-separated bounding box coordinates per line in the format `(x1, y1, x2, y2)`.

## Outputs

Results are saved to the `--output_file` in the format:

```
<image_filename> <detected_face_count>
```

If ground truth bounding boxes are provided, the script evaluates the detections with precision, recall, and F1 score, displayed in the console.

If a ground truth labels file is provided, the script calculates and displays the overall accuracy.

## Test Script

Use `unittest_face_detection.py` to test the `read_data` function in the `script.py` module. Run the test with:

```
python unittest_face_detection.py
```

## Notes

Processing many images can be time-consuming. Ensure sufficient disk space is available for the outputs.

