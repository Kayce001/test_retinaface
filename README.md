# RetinaFace Image Assessment

This project uses RetinaFace, a face detection model, to process images in a specified folder and reports how many faces are detected in each image. You can optionally compare the detected count to a ground truth file, and compute the accuracy of the face detection.

## Dependencies

- Python 3.10+
- OpenCV
- PyTorch
- NumPy
- Ubuntu 20.04+

## Installation

There are two methods to set up the environment for this project:

### Method 1: Using Conda Environment File

To install using the provided `environment.yml` file within the `test_retinaface` directory, follow these steps:

1. Clone the Pytorch_Retinaface repository: 
   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```
2. Navigate into the cloned directory:
   ```
   cd Pytorch_Retinaface
   ```
3. Copy the project's `environment.yml`、`unittest_face_detection.py` and `script.py` files into the `Pytorch_Retinaface` directory.
   
4. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate retinaface
   ```
5. Download and install the RetinaFace model to the `./weights` folder. Just follow the instructions provided in the Pytorch_Retinaface repository.

### Method 2: Manual Installation

1. Confirm you have a compatible Python version installed (3.10+ required) and you're running Ubuntu 20.04 or later.
2. Clone the Pytorch_Retinaface repository: 
   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```
3. Navigate into the cloned directory: 
   ```
   cd Pytorch_Retinaface
   ```
4. Copy the project's `requirements.txt`、`unittest_face_detection.py` and `script.py` files into the `Pytorch_Retinaface` directory.

5. Install the required Python libraries: 
   ```
   pip install -r requirements.txt
   ```
6. Download and install the RetinaFace model to the `./weights` folder. Just follow the instructions provided in the Pytorch_Retinaface repository.


## Usage

Run the script with the following command:

```
python script.py --input_folder /path/to/images --output_file /path/to/output_file --truth /path/to/truth_file --ground_truth /path/to/ground_truth_folder --m /path/to/model [--network <network>] [--confidence_threshold <confidence_threshold>] [--nms_threshold <nms_threshold>] [--cpu]
```

Options:

- `--m`: Path to the trained state_dict file (default: './weights/mobilenet0.25_Final.pth').
- `--network`: Choice of backbone network. Options are: 'mobile0.25' or 'resnet50' (default: 'mobile0.25').
- `--cpu`: Use this flag to enable CPU inference (default: False).
- `--confidence_threshold`: Confidence threshold for face detection (default: 0.6).
- `--nms_threshold`: Non-maximum suppression threshold (default: 0.92).
- `--input_folder`: Path to the folder containing the images to process.
- `--output_file`: Path to the output file where the results will be written (default: 'output.txt').
- `--truth`: Path to the file containing ground truth labels for the face counts per image (optional).
- `--ground_truth`: Path to the folder containing ground truth bounding boxes (optional).

The `--truth` file should contain space-separated values, with one row per image file. Each row should have the image filename and the true number of faces in that image:

```
image1.jpg 3
image2.jpg 1
image3.jpg 0
...
```

The `--ground_truth` folder should contain `.txt` files with the same basename as the image files. Each `.txt` file should contain space-separated values representing the bounding boxes in the format (x1, y1, x2, y2), one bounding box per line.

## Outputs

Outputs are written to the specified output file. Each line in the output file represents one image with the following format:

```
<image_filename> <detected_face_count>
```

Additionally, if the ground truth bounding boxes are provided, the script will evaluate the detection results using precision, recall, and F1 score, and print them to the console:

```
Precision: XX.XX%
Recall: XX.XX%
F1 Score: XX.XX%
```

If a ground truth labels file is provided, the script will also calculate and print the overall accuracy:

```
Accuracy: XX.XX%
```
```


## Test Script

The `unittest_face_detection.py` script is designed to test the `read_data` function within the `script` module. This function is expected to correctly read and parse a file containing image filenames along with the number of faces detected in each image. The script verifies that the function returns a dictionary with image filenames as keys and the number of faces as corresponding values. Automated tests like this are crucial for ensuring the reliability and accuracy of face detection software over different updates and changes in the codebase.

To execute the test script, run the following command:
```
python unittest_face_detection.py
```

## Notes

This script can take a while to process a large amount of images. Make sure you have enough disk space to store the output results.

