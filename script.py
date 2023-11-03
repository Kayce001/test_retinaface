
import argparse
import torch
import cv2
import numpy as np
import glob
import os
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm



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

args = parser.parse_args()  

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
    return len(dets)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    if args.network == 'resnet50':
        cfg = cfg_re50
    elif args.network == 'mobile0.25':
        cfg = cfg_mnet
    else:
        print("Unsupported network configuration")
        exit()
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    if not args.cpu:
        if torch.cuda.is_available():
            net = net.to('cuda')
    net.eval()

    ground_truth = {}
    if args.truth is not None:
        ground_truth = read_data(args.truth)
        
    correct_predictions = 0
    total_predictions = 0

    # Ensure the output file is open
    with open(args.output_file, 'w') as fw:
        print("ready")
        # Walk through the input folder and run face detection on all image files
        image_paths = glob.glob(os.path.join(args.input_folder, '*'))
        for image_path in image_paths:
            num_faces = detect_faces(net, image_path)
            image_filename = os.path.basename(image_path)  # Extract the filename from the path
            fw.write("{} {}\n".format(image_filename, num_faces))

            if args.truth is not None and image_filename in ground_truth:
                if num_faces == ground_truth[image_filename]:
                    correct_predictions += 1
                total_predictions += 1

    if args.truth is not None:
        accuracy = correct_predictions / total_predictions
        print(f'Accuracy: {accuracy * 100.0 :.2f}%')

    print('Finished processing images and results are written to output file!')