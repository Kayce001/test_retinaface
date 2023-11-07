# Import necessary libraries
import argparse
import torch
import cv2
import numpy as np
import glob
import os
# Import necessary modules from the local project structure
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode



parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
parser.add_argument('--nms_threshold', default=0.92, type=float, help='nms_threshold')
parser.add_argument('--input_folder', type=str, help='Folder containing the images to detect')
parser.add_argument('--output_file', type=str, default='output.txt', help='File to write the output')
parser.add_argument('--truth', type=str, help='Path to the file containing ground truth labels')
parser.add_argument('--ground_truth', type=str, help='Folder containing ground truth bounding boxes')


args = parser.parse_args()  

# Define function to check and reconcile pretrained model keys and the model state dict keys
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    return unused_pretrained_keys, used_pretrained_keys, missing_keys

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)

    # Now call the functions to clean 'state_dict' and update the model's state_dict
    unused, used_keys, missing_keys = check_keys(model, pretrained_dict)
    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# Define function to read ground truth bounding box annotations from text files
def read_ground_truth_bounding_boxes(gt_folder, image_filename):
    # Split the extension and replace it with .txt
    base_filename = os.path.splitext(image_filename)[0] + '.txt'
    gt_path = os.path.join(gt_folder, base_filename)
    gt_bounding_boxes = []
    if os.path.isfile(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    # Convert string to float and store it as a tuple (x1, y1, x2, y2)
                    gt_bounding_boxes.append(tuple(map(float, parts)))
    return gt_bounding_boxes

# Define function to read data for accuracy comparison from a truth file
def read_data(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            data_dict[parts[0]] = int(parts[1])
    return data_dict

def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum([y_true[i] == y_pred[i] for i in y_true])
    return correct_predictions / len(y_true)

# Define a function to calculate the Intersection over Union (IoU) for two bounding boxes
def iou(box_a, box_b):
    # Compute the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and ground truth rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

# Define function to update detection metrics (true positives, false positives, false negatives)
def update_metrics(detected_boxes, gt_boxes, iou_threshold=0.5):
    TP = FP = FN = 0
    matched_gt_boxes = []

    # Check each detected box against all ground truth boxes
    for det_box in detected_boxes:
        match_found = False
        for gt_box in gt_boxes:
            if iou(det_box, gt_box) >= iou_threshold:
                if gt_box not in matched_gt_boxes:
                    match_found = True
                    matched_gt_boxes.append(gt_box)
                    break
        if match_found:
            TP += 1
        else:
            FP += 1

    # Any ground truth box not matched is a false negative
    FN = len(gt_boxes) - len(matched_gt_boxes)

    return TP, FP, FN

def load_truth_data(truth_path):
    truth_data = {}
    if os.path.isfile(truth_path):
        with open(truth_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    truth_data[parts[0]] = int(parts[1])
    return truth_data

# Define the main face detection function which uses the RetinaFace model
def detect_faces(net, image_path):
    resize = 1
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    if not args.cpu:
        if torch.cuda.is_available():
            img = img.to('cuda')
    loc, conf, _ = net(img)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    
    priors = priorbox.forward()
    priors = priorbox.forward()
    if not args.cpu:
        if torch.cuda.is_available():
            priors = priors.to('cuda')
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    if not args.cpu:
        if torch.cuda.is_available():
            boxes = boxes.to('cuda')
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    dets = dets[keep, :]
    return dets

# Main execution block
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    
    if args.network == 'resnet50':
        cfg = cfg_re50
    elif args.network == 'mobile0.25':
        cfg = cfg_mnet
    else:
        print("Unsupported network configuration")
        exit()

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    if not args.cpu:
        net = net.cuda()
    net.eval()

    # Initialize metrics
    TP = FP = FN = 0
    correct_predictions = 0
    total_predictions = 0

    truth_data = {}
    if args.truth:
        truth_data = load_truth_data(args.truth)
        
    # Open the output file to write detection results
    with open(args.output_file, 'w') as fw:
        print("Ready to process images...")
        image_paths = glob.glob(os.path.join(args.input_folder, '*'))
        for image_path in image_paths:
            detected_boxes = detect_faces(net, image_path)  # Assumes this returns bounding boxes
            image_filename = os.path.basename(image_path)
            fw.write(f"{image_filename} {len(detected_boxes)}\n")
            
            # If ground truth annotations are provided, read them and update evaluation metrics
            if args.ground_truth is not None:
                gt_boxes = read_ground_truth_bounding_boxes(args.ground_truth, image_filename)
                tp, fp, fn = update_metrics(detected_boxes, gt_boxes)
                TP += tp
                FP += fp
                FN += fn

            # If truth data for accuracy calculation is provided, use it
            if args.truth is not None:
                true_count = truth_data.get(image_filename, 0)
                if len(detected_boxes) == true_count:
                    correct_predictions += 1
                total_predictions += 1

    if args.ground_truth is not None:
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1_score:.4f}')
    
    if args.truth is not None and total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f'Accuracy: {accuracy * 100.0:.2f}%')

    print('Finished processing images. The results are written to the output file.')
