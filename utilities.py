import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from tqdm import tqdm
import rasterio
import pandas as pd
import os
import shutil
import torch

def filterDataset(filename):
    with open('dataset_filter/' + filename + '.txt', 'r') as f:
        data_files = [line.strip() for line in f if line.strip()]
    return data_files

def plot_one_prediction(img_array, titles=None, ncol=3):
    """
    plot (RGB image | Ground Truth Mask | Predicted Mask)
    Args:
        img_array (numpy array)
    """
    nrow = len(img_array) // ncol
    _, plots = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 4))

    if nrow == 1:
        plots = np.expand_dims(plots, axis=0)

    for i in range(len(img_array)):
        row = i // ncol
        col = i % ncol
        ax = plots[row, col]
        img = img_array[i]
        if img.ndim == 2:
            img[0][0] = 0 # for mask all 1s
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i], fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_predictions(rgb_images, target_masks, pred_masks):
    img_arrays = []
    titles = []

    for i in range(len(rgb_images)):
        rgb = rgb_images[i]
        target = np.squeeze(target_masks[i])
        pred = np.squeeze(pred_masks[i])

        img_arrays.extend([rgb, target, pred])
        titles.extend([f'RGB {i+1}', f'Target {i+1}', f'Prediction {i+1}'])

    plot_one_prediction(img_arrays, titles=titles, ncol=3)

def getMeanAndStd(dataset):
    """
    Get mean and standard deviation of a dataset
    """

    df = pd.read_csv('dataset_summary.csv')
    df = df[df['image_file'].str.contains('|'.join(dataset), case=False, na=False)]
    
    mean = [
        df['mean_red'].mean(),
        df['mean_green'].mean(),
        df['mean_blue'].mean(),
        df['mean_ir'].mean()
    ]

    std = [
        df['std_red'].mean(),
        df['std_green'].mean(),
        df['std_blue'].mean(),
        df['std_ir'].mean()
    ]
    return np.array(mean), np.array(std)

def reverse_transform(inp, mean, std):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean

    return inp

class LazyDataset(Dataset):
    def __init__(self, image_files, input_dir='dataset', transform=None):
        self.image_files = image_files
        self.input_dir = input_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        image_path = os.path.join(self.input_dir, 'data', file)
        mask_path = os.path.join(self.input_dir, 'masks', file)

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
            image = torch.from_numpy(image)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # (H, W)
            mask = torch.from_numpy(mask).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, mask
    
def split_dataset(file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    df = pd.read_csv('dataset_summary.csv')
    cloud_coverage = df['cloud_coverage']
    cf =  df[cloud_coverage == 0]['image_file'].tolist()
    fc =  df[cloud_coverage == 1]['image_file'].tolist()

    cloud_free = []
    partially_clouded = []
    fully_clouded = []

    for f in file_list:
        if f in cf: 
            cloud_free.append(f)
        elif f in fc:
            fully_clouded.append(f)
        else:
            partially_clouded.append(f)

    total_cf = len(cloud_free)
    train_end_cf = int(total_cf * train_ratio)
    val_end_cf = train_end_cf + int(total_cf * val_ratio)

    total_fc = len(fully_clouded)
    train_end_fc = int(total_fc * train_ratio)
    val_end_fc = train_end_fc + int(total_fc * val_ratio)

    total_pc = len(partially_clouded)
    train_end_pc = int(total_pc * train_ratio)
    val_end_pc = train_end_pc + int(total_pc * val_ratio)

    train_files = cloud_free[:train_end_cf] + partially_clouded[:train_end_pc] + fully_clouded[:train_end_fc]
    val_files = cloud_free[train_end_cf : val_end_cf] + partially_clouded[train_end_pc : val_end_pc] + fully_clouded[train_end_fc : val_end_fc]
    test_files = cloud_free[val_end_cf:] + partially_clouded[val_end_pc:] + fully_clouded[val_end_fc:]

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    return train_files, val_files, test_files

def get_data_loader(data_files, trans, input_dir='dataset', batch_size=25, shuffle=True):

    data_set = LazyDataset(data_files, input_dir=input_dir, transform=trans)
    
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        
    return dataloader

def calc_pos_weight(image_files):
    
    df = pd.read_csv('dataset_summary.csv')
    df = df[df['image_file'].str.contains('|'.join(image_files), case=False, na=False)]['cloud_coverage']
    pos = df.sum()
    neg = df.count() - pos

    return torch.tensor([neg / (pos + 1e-6)], dtype=torch.float32)

def dice_coefficient(pred, target, smooth=1e-7, val=False, th=0.5):
    if val:
        pred = (pred > th).float()
        target = target.float()

    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    pred_sum = pred.sum(dim=(2, 3))
    target_sum = target.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice.mean()

def jaccard_index(pred, target, smooth=1., val=False, th=0.5):
    if val:
        pred = (pred > th).float()
        target = target.float()

    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    total = (pred + target).sum(dim=(2, 3))
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.mean()

def calc_loss(pred, target, metrics, bce_weight=None, val=False, th=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=bce_weight)

    pred = F.sigmoid(pred)
    
    dice = dice_coefficient(pred, target, val=val, th=th)
    jaccard = jaccard_index(pred, target, val=val, th=th)
    loss = bce * 0.5 + (1 - dice_coefficient(pred, target)) * 0.5
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    metrics['jaccard'] += jaccard.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_one_epoch(model, dataloader, optimizer, device, pos_weight, epoch_num=None, num_epochs=None):
    model.train()
    metrics = defaultdict(float)
    epoch_samples = 0
    dataloader = tqdm(dataloader, desc=f"Train Phase Epoch {epoch_num}/{num_epochs}", leave=False)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = calc_loss(outputs, labels, metrics, bce_weight=pos_weight)
            loss.backward()
            optimizer.step()

        epoch_samples += inputs.size(0)
        dataloader.set_postfix(loss=(metrics['loss'] / epoch_samples))

    return metrics['loss'] / epoch_samples

def validate_one_epoch(model, dataloader, device, thresholds):
    model.eval()
    best_threshold = None
    best_loss = float('inf')
    best_dice = 0

    with torch.no_grad():
        for threshold in thresholds:
            metrics = defaultdict(float)
            epoch_samples = 0
            dataloader = tqdm(dataloader, desc=f"Validation | Threshold {threshold:.2f}", leave=False)

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                calc_loss(outputs, labels, metrics, val=True, th=threshold)
                epoch_samples += inputs.size(0)
                dataloader.set_postfix(loss=(metrics['loss'] / epoch_samples), dice=(metrics['dice'] / epoch_samples))

            avg_loss = metrics['loss'] / epoch_samples
            avg_dice = metrics['dice'] / epoch_samples
            # print(f"Validation Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")

            if avg_dice > best_dice:
                best_loss = avg_loss
                best_dice = avg_dice
                best_threshold = threshold

    print(f"Best threshold: {best_threshold:.2f} with loss: {best_loss:.4f} with Dice: {best_dice:.4f}")
    return best_dice, best_threshold

def train_model(model, optimizer, scheduler, dataloaders, pos_weight, num_epochs=25, patience=5):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pos_weight = pos_weight.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = 0
    best_threshold = 0.5
    thresholds = np.linspace(0.2, 0.8, 20)
    patience_counter = 0

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(model, dataloaders['train'], optimizer, device, pos_weight, epoch_num=epoch+1, num_epochs=num_epochs)
        val_dice, val_threshold = validate_one_epoch(model, dataloaders['val'], device, thresholds)
        scheduler.step(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_threshold = val_threshold
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save model to file
            torch.save(model.state_dict(), 'outputs/temp/best_model.pkl')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"No Improvement for {patience} epochs, Early Stopping...")
            break
    
    print(f'Best val dice: {best_dice:.4f}, Best Threshold: {best_threshold:.2f}')

    # Write threshold to file
    with open("outputs/temp/best_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    model.load_state_dict(best_model_wts)

    return model, best_threshold

def run(UNet, lr= 1e-3, num_epochs= 60, dataset_filter='cleaned', patience=10, gamma=0.5, batch_size=4):
    
    data_files = filterDataset(dataset_filter)
    
    mean, std = getMeanAndStd(data_files)

    transform = transforms.Compose([transforms.Normalize(mean, std)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet.to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=5, factor=gamma)

    train_files, val_files, test_files = split_dataset(data_files)

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_dataloader = get_data_loader(train_files, trans=transform, batch_size=batch_size)
    val_dataloader = get_data_loader(val_files, trans=transform, batch_size=batch_size)
    test_dataloader = get_data_loader(test_files, trans=transform, batch_size=batch_size)

    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
    }

    pos_weight = calc_pos_weight(data_files)
   
    model, best_threshold = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, pos_weight, num_epochs=num_epochs, patience=patience)
      
    validate_one_epoch(model, test_dataloader, device, thresholds=[best_threshold])
    
def predict_one(model, image_path, mask_path, threshold=0.5, visualize=False, dataset_filter='dataset'):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    with rasterio.open(image_path) as src:
        image = src.read()  # shape: (C, H, W)
    image = image.astype(np.float32)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Read the first band only
    mask = mask.astype(np.uint8)
    mask[0][0] = 0 # for plotting

    data_files = filterDataset(dataset_filter)

    mean, std = getMeanAndStd(data_files)
    transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])

    image_tensor = torch.from_numpy(image)
    image_tensor = transform(image_tensor)
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension  # shape: (1, C, H, W)
    image_tensor = image_tensor.to(device)

    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)
    mask = mask.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > threshold).float()

    matching_pixels = (pred == mask).sum().item()
    total_pixels = torch.numel(mask)

    percentage_match = (matching_pixels / total_pixels) * 100

    if visualize:
        print(f"Match: {percentage_match:.2f}")

    pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    pred = pred_mask

    if visualize:
        pred_mask[0][0] = 0 # for plotting
        image_rgb = reverse_transform(torch.from_numpy(image.copy()), mean, std)[..., :3]  # Take RGB
        image_rgb = image_rgb / image_rgb.max()  # Normalize for display

        # Plotting
        _, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(image_rgb)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        axs[1].imshow(mask.cpu().squeeze(0), cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis("off")

        axs[2].imshow(pred_mask, cmap='gray')
        axs[2].set_title("Predicted Mask")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

    return percentage_match, pred

def predict_and_clean(model, dataset_filter='dataset', best_threshold=0.5, confidence=90, replace=False, visualize=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    data_files = filterDataset(dataset_filter)

    total_samples = 0
    total_percentage = 0

    noise = []
    cleaned = []

    for file_name in tqdm(data_files, desc="Predicting"):

        image_path = 'dataset' + '/data/' + file_name
        mask_path = 'dataset' + '/masks/' + file_name

        percentage_match, pred = predict_one(model, image_path, mask_path, threshold=best_threshold, visualize=visualize, dataset_filter=dataset_filter)
        
        total_percentage += percentage_match
        total_samples += 1

        # print(f"Percentage Match: {percentage_match}")

        if percentage_match < confidence:
            if replace:
                shutil.copy(os.path.join('dataset' + '/data', file_name), os.path.join('inputs/processed/data', file_name))
                save_predicted_mask(pred_mask=pred, 
                                    reference_tif_path='dataset/masks/100096.tif', save_folder='inputs/processed/masks', save_name=file_name)
            else:
                noise.append(file_name)
        else:
            cleaned.append(file_name)

    avg_percentage_match = total_percentage / total_samples
    print(f"Average Percentage Match on Test Set: {avg_percentage_match:.2f}%")
    return cleaned, noise

def save_predicted_mask(pred_mask: np.ndarray, reference_tif_path: str, save_folder: str, save_name: str):
    
    os.makedirs(save_folder, exist_ok=True)
    
    with rasterio.open(reference_tif_path) as src:
        meta = src.meta.copy()
    
    # Ensure correct shape
    if pred_mask.ndim == 2:
        pred_mask = np.expand_dims(pred_mask, 0)  # (1, H, W)

    meta.update({
        "count": 1,  # number of channels
        "dtype": rasterio.uint8,
    })

    save_path = os.path.join(save_folder, save_name)
    with rasterio.open(save_path, 'w', **meta) as dst:
        dst.write(pred_mask.astype(np.uint8))

    # print(f"Saved predicted mask to: {save_path}")

def inference(model, image_path, threshold, transform):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    with rasterio.open(image_path) as src:
        image = src.read()  # shape: (C, H, W)
    image = image.astype(np.float32)

    image_tensor = torch.from_numpy(image)
    image_tensor = transform(image_tensor)
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension  # shape: (1, C, H, W)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > threshold).float()

    pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8)
    return pred_mask

