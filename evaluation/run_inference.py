import utilities
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import os
import rle_encoder_decoder
import sys
from torchvision import transforms
from unet import UNet
from skimage.transform import resize

def generateSubmission(input_dir='test'):

    model = UNet()
    model.load_state_dict(torch.load('models/unet_model.pkl'))
    print('Model Loaded Successfully')

    threshold = 0.5
    with open('thresholds/unet_best_threshold.txt', 'r') as f:
        threshold = float(f.read())
    print('Threshold Loaded Successfully')

    test_files = [f for f in os.listdir(input_dir)]

    data_files = [f for f in os.listdir('inputs/cleaned/data')]

    mean, std = utilities.getMeanAndStd(data_files)
    transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])

    segmentations = []
    ids = []
    for f in tqdm(test_files, desc="Generating Submission"):

        id = os.path.splitext(f)[0]

        img_path = input_dir + '/' + f

        pred = utilities.inference(model, img_path, threshold, transform)
        pred = resize(pred, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(pred.dtype)

        encoded_pred = rle_encoder_decoder.rle_encode(pred)

        segmentations.append(encoded_pred)
        ids.append(id)

    submission = pd.DataFrame({
        'id' : ids,
        'segmentation': segmentations
    })

    submission.to_csv('outputs/submission.csv')

    return submission

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise Exception("Test folder name not provided. Usage: python run_inference.py <test_folder_name>")
    
    submission_df = generateSubmission(sys.argv[1]) 
    