import utilities
import torch
from tqdm import tqdm
import pandas as pd
import os
from evaluation import rle_encoder_decoder
import sys
from unet import UNet
from skimage.transform import resize
import numpy as np

def generateSubmission(input_dir='test', model_file_name='models/instance_norm_e3.pkl'):

    model = UNet()
    model.load_state_dict(torch.load(model_file_name))
    print('Model Loaded Successfully')

    threshold = 0.5
    with open('thresholds/unet_best_threshold.txt', 'r') as f:
        threshold = float(f.read())
    print('Threshold Loaded Successfully')

    test_files = [f for f in os.listdir(input_dir)]

    submissions = []
    for f in tqdm(test_files, desc="Generating Submission"):

        id = os.path.splitext(f)[0]

        img_path = input_dir + '/' + f

        pred = utilities.inference(model, img_path, threshold)
        pred = resize(pred, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

        encoded_pred = rle_encoder_decoder.rle_encode(pred)

        submissions.append({
            "id": id,
            "segmentation": encoded_pred
        })

    df = pd.read_csv("sample_submission.csv", dtype={'id': str})

    for submission in submissions:
        df.loc[df['id'] == submission['id'], 'segmentation'] = submission['segmentation']
    df.to_csv('submission.csv', index=False)
    print(f"Saved submission to submission.csv")

    return submission

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        raise Exception("Usage: python run_inference.py <test_folder_name> <model_file_name>")
    
    submission_df = generateSubmission(sys.argv[1], sys.argv[2]) 
    