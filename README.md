# RetinaFace Image Assessment

This project uses RetinaFace, a face detection model, to process images in a specified folder and reports how many faces are detected in each image. You can optionally compare the detected count to a ground truth file, and compute the accuracy of the face detection.

## Dependencies

- Python 3.10+
- OpenCV
- PyTorch
- NumPy
- Ubuntu 20.04+

## Installation

1. Confirm you have a compatible Python version installed (3.10+ required) and you're running Ubuntu 20.04 or later.

2. Clone the Pytorch_Retinaface repository: 
   ```
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   ```
   
3. Navigate into the cloned directory: 
   ```
   cd Pytorch_Retinaface
   ```

4. Copy the project's `requirements.txt` and `script.py` files into the `Pytorch_Retinaface` directory.

5. Install the required Python libraries: 
   ```
   pip install -r requirements.txt
   ```

6. Download and install the RetinaFace model to the `./weights` folder. Just follow the instructions provided in the Pytorch_Retinaface repository.

## Usage

```
python script.py --input_folder /path/to/images --output_file /path/to/output_file --truth /path/to/truth_file [--m <model_path>] [--network <network>] [--confidence_threshold <confidence_threshold>] [--nms_threshold <nms_threshold>] [--cpu]
```

Options:

- `--m`: Path to the trained state_dict file (default: './weights/mobilenet0.25_Final.pth').
- `--network`: Choice of backbone network. Options are: 'mobile0.25' or 'resnet50' (default: 'mobile0.25').
- `--cpu`: Use this flag to enable cpu inference (default: False).
- `--confidence_threshold`: Confidence threshold for face detection (default: 0.6).
- `--nms_threshold`: Non-maximum suppression threshold (default: 0.92).
- `--input_folder`: Path to the folder containing the images to process.
- `--output_file`: Path to the output file where the results will be written (default: 'output.txt').
- `--truth`: Path to the file containing ground truth labels for the face counts per image (optional).

The `--truth` file should be a text file of space-separated values, with one row per image file. Each row should contain the image filename and the true number of faces in that image. Example:

```
image1.jpg 3
image2.jpg 1
image3.jpg 0
...
```

## Outputs

Outputs are written to the provided output file (default 'output.txt'). Each line represents an image, with the format:

```
<image_filename> <detected_face_count>
```

If a ground truth file is provided, the script will also print the detection accuracy to the console:

```
Accuracy: XX.XX%
```

## Notes

This script can take a while to process a large amount of images. Make sure you have enough disk space to store the output results.
