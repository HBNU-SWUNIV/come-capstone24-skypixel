import os
import numpy as np
import cv2
import torch
import argparse
from torch.utils.data import DataLoader, Dataset as BaseDataset
import segmentation_models_pytorch as smp
import albumentations as albu

# Argument parser
parser = argparse.ArgumentParser(description='Inference on test dataset and save predictions')
parser.add_argument('--test_dir', type=str, default='',
                    help='Directory containing test images')
parser.add_argument('--model_path', type=str, default='',
                    help='Path to the saved model')
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Directory where predictions will be saved')
parser.add_argument('--resize', type=int, default=256,
                    help='Resize dimension for images and masks')
parser.add_argument('--batch', type=int, default=8,
                    help='Batch size for inference')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device number')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Test dataset class without labels
class TestDataset(BaseDataset):
    def __init__(self, image_dir, augmentation=None, preprocessing=None):
        self.image_fps = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.imread(self.image_fps[i], cv2.IMREAD_COLOR)

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        
        return image, self.image_fps[i]  # Return image and file path
    
    def __len__(self):
        return len(self.image_fps)

# Data augmentation and preprocessing
def get_validation_augmentation():
    test_transform = [
        albu.Resize(args.resize, args.resize, p=1.0),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    return albu.Compose([
        albu.Lambda(image=to_tensor),
    ])

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Load the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = 'tu-efficientnet_b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
ACTIVATION = 'sigmoid'

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# 모델을 병렬처리로 로드
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for inference")
    model = torch.nn.DataParallel(model)

# 모델 상태를 로드
state_dict = torch.load(args.model_path)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # 'module.' 제거
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)

model = model.to(DEVICE)
model.eval()

# Preprocessing function based on the encoder used during training
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Load test dataset
test_dataset = TestDataset(
    args.test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn)
)

test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

# Function to convert prediction mask to binary image format
def mask_to_grayscale(mask):
    """Convert a binary mask to a grayscale image (0 or 255 values)."""
    mask = mask.squeeze().cpu().numpy()  # Convert to numpy and remove extra dimensions
    mask = (mask > 0.6).astype(np.uint8) * 255  # Threshold and convert to uint8
    return mask

# Inference and save predictions
with torch.no_grad():
    for images, file_paths in test_loader:
        images = images.to(DEVICE)
        predictions = model(images)
        
        # Apply sigmoid and threshold to get binary masks
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.6).float()

        # Save each prediction
        for pred, file_path in zip(predictions, file_paths):
            pred_grayscale = mask_to_grayscale(pred)  # Convert prediction to grayscale image
            save_path = os.path.join(args.output_dir, os.path.basename(file_path))
            cv2.imwrite(save_path, pred_grayscale)  # Save the binary prediction as an image
            print(f'Saved prediction: {save_path}')
