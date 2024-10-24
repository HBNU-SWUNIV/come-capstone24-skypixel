import os
import numpy as np
import pandas as pd
import cv2
import warnings
import ssl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch.utils as utils
import segmentation_models_pytorch.losses as losses
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from segmentation_models_pytorch.utils.metrics import IoU
from tqdm import tqdm
import albumentations as albu
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import argparse
import wandb

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Train and validate a model with directory paths.')
parser.add_argument('--train_csv', type=str, default='',
                    help='Path to the training CSV file')
parser.add_argument('--valid_csv', type=str, default='',
                    help='Path to the validation CSV file')
parser.add_argument('--batch', type=int, default=8,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--resize', type=int, default=256,
                    help='Resize dimension for images and masks')
parser.add_argument('--project', type=str, default='SAR_seg',
                    help='WandB project name')
parser.add_argument('--dataset', type=str, default='',
                    help='Dataset name')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of workers for data loading')
parser.add_argument('--method', type=str, default='baseline',
                    help='Training method')
parser.add_argument('--gpu', type=str, default='0',
                    help='GPU device number')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
name = f'{args.dataset}_{args.batch}_{args.epochs}_{args.lr}_{args.method}_{args.resize}'
wandb.init(project=args.project, name=name, config=vars(args))

train_csv_path = args.train_csv
valid_csv_path = args.valid_csv
num_workers = args.num_workers

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

workspace_path = '/data/com_0/LJS/SAR_Segmentation/'
segmentation_path = os.path.join(workspace_path, 'segmentation_models')

class Dataset(BaseDataset):
    
    CLASSES = ['building']
    
    def __init__(self, csv_file, classes=None, augmentation=None, preprocessing=None):
        self.data = pd.read_csv(csv_file)  # CSV 파일 로드
        self.images_fps = self.data['image'].tolist()  # 이미지 경로 리스트
        self.masks_fps = self.data['label'].tolist()  # 마스크 경로 리스트

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        
        mask = (mask > 127).astype('float32')  
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)

def get_training_augmentation():
    train_transform = [
        albu.RandomCrop(args.resize, args.resize, p=1.0),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.CenterCrop(args.resize, args.resize, p=1.0),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [
        #albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

ENCODER = 'tu-efficientnet_b0'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['building']
ACTIVATION = 'sigmoid' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
).to(DEVICE)

# DataParallel 적용
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    train_csv_path, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

valid_dataset = Dataset(
    valid_csv_path, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES
)

train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, num_workers=num_workers)

class CombinedLoss(nn.Module):
    def __init__(self, loss_a, loss_b, weight_a=0.5, weight_b=0.5):
        super(CombinedLoss, self).__init__()
        self.loss_a = loss_a
        self.loss_b = loss_b
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.__name__ = loss_a.__class__.__name__ + '_' + loss_b.__class__.__name__ 

    def forward(self, output, target):
        return self.weight_a * self.loss_a(output, target) + self.weight_b * self.loss_b(output, target)

class DiceScore(base.Metric):
    __name__ = "dice_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=1.0,  
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

DiceLoss = utils.losses.DiceLoss()
CE_Loss = torch.nn.CrossEntropyLoss()
combined_criterion = CombinedLoss(DiceLoss, CE_Loss, weight_a=0.5, weight_b=0.5)

metrics = [
    DiceScore(),
    IoU(),
]

optimizer = torch.optim.AdamW([
    dict(params=model.parameters(), lr=args.lr, weight_decay=0.001),
])

def mask_to_grayscale(mask):
    """Convert a binary mask to Grayscale."""
    grayscale_mask = (mask * 255).astype(np.uint8)
    return grayscale_mask

# 스케줄러 추가
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

total_steps = len(train_loader) * args.epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

# Train epoch with real-time loss display
class TrainEpochWithProgressBar(utils.train.TrainEpoch):
    def run(self, dataloader):
        self.model.train()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        images, masks, preds = [], [], []

        with tqdm(total=len(dataloader), desc='Train', unit='batch') as pbar:
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(x)
                loss = self.loss(prediction, y)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(prediction, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                pbar.set_postfix(loss=loss_value, lr=self.optimizer.param_groups[0]['lr'])
                pbar.update()

                # Save images and predictions for the first 15 batches
                if len(images) < 15:
                    images.extend(x.cpu())
                    masks.extend(y.cpu())
                    preds.extend(prediction.cpu())

                # 스케줄러 업데이트
                scheduler.step()

        logs['loss'] = loss_meter.mean
        for metric_name, meter in metrics_meters.items():
            logs[metric_name] = meter.mean

        return logs, images, masks, preds

# Valid epoch with real-time loss display and logging
class ValidEpochWithLogging(utils.train.ValidEpoch):
    def run(self, dataloader):
        self.model.eval()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        images, masks, preds = [], [], []

        with torch.no_grad():
            with tqdm(total=len(dataloader), desc='Valid', unit='batch') as pbar:
                for x, y in dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    prediction = self.model(x)
                    loss = self.loss(prediction, y)

                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)

                    for metric_fn in self.metrics:
                        metric_value = metric_fn(prediction, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)

                    pbar.set_postfix(loss=loss_value)
                    pbar.update()

                    # Save images and predictions for the first 15 batches
                    if len(images) < 15:
                        images.extend(x.cpu())
                        masks.extend(y.cpu())
                        preds.extend(prediction.cpu())

        logs['loss'] = loss_meter.mean
        for metric_name, meter in metrics_meters.items():
            logs[metric_name] = meter.mean

        return logs, images, masks, preds

def log_images_to_wandb(epoch, train_images, train_masks, train_preds, valid_images, valid_masks, valid_preds):
    log_images = []

    # Valid images
    for img, mask, pred in zip(valid_images[:15], valid_masks[:15], valid_preds[:15]):
        img_np = img.permute(1, 2, 0).numpy().astype(np.uint8)
        mask_np = mask.numpy().squeeze()
        pred_np = torch.sigmoid(pred).detach().numpy().squeeze()  # detach() 추가

        # 이진화 수행
        pred_np = (pred_np > 0.5).astype(np.uint8) * 255

        mask_gray = mask_to_grayscale(mask_np)
        pred_gray = mask_to_grayscale(pred_np)

        log_images.append(wandb.Image(img_np, caption="Valid Image"))
        log_images.append(wandb.Image(mask_gray, caption="Valid Label"))
        log_images.append(wandb.Image(pred_gray, caption="Valid Prediction"))

    wandb.log({
        "epoch": epoch + 1,
        "Images": log_images,
        "Train Loss": train_logs['loss'],
        "Valid Loss": valid_logs['loss'],
        "Train Dice Score": train_logs['dice_score'],
        "Valid Dice Score": valid_logs['dice_score'],
        "Train IoU Score": train_logs['iou_score'],
        "Valid IoU Score": valid_logs['iou_score'],
        "Learning Rate": optimizer.param_groups[0]['lr']
    })

train_epoch = TrainEpochWithProgressBar(
    model, 
    loss=combined_criterion, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = ValidEpochWithLogging(
    model, 
    loss=combined_criterion, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

epochs = args.epochs
save_interval = 10  

max_score = 0

save_dir = os.path.join(workspace_path, 'ckpt')
os.makedirs(save_dir, exist_ok=True)

config = wandb.config

train_cumulative_dice_score = 0.0
train_cumulative_loss = 0.0

max_dice_score = 0

best_model_path = None

for epoch in range(epochs):
    print('\nEpoch: {}'.format(epoch + 1))

    # Training phase
    train_logs, train_images, train_masks, train_preds = train_epoch.run(train_loader)
    
    # Validation phase
    valid_logs, valid_images, valid_masks, valid_preds = valid_epoch.run(valid_loader)

    # Print dice and miou for each epoch
    print(f"Epoch {epoch + 1} - Train Loss: {train_logs['loss']}, Train Dice Score: {train_logs['dice_score']}, Train IoU Score: {train_logs['iou_score']}")
    print(f"Epoch {epoch + 1} - Valid Loss: {valid_logs['loss']}, Valid Dice Score: {valid_logs['dice_score']}, Valid IoU Score: {valid_logs['iou_score']}")

    # Log results and images to WandB for each epoch
    log_images_to_wandb(epoch, train_images, train_masks, train_preds, valid_images, valid_masks, valid_preds)

    # Save the model with the highest Dice score
    if valid_logs['dice_score'] > max_dice_score:
        print('Improved Dice Score from {} to {}. Saving model...'.format(max_dice_score, valid_logs['dice_score']))
        max_dice_score = valid_logs['dice_score']
        new_model_path = os.path.join(save_dir, f'best_model_{args.dataset}_epoch_{epoch}.pth')
        torch.save(model.state_dict(), new_model_path)
        
        # 이전 최고 성능의 모델 삭제
        if best_model_path is not None:
            os.remove(best_model_path)
        
        best_model_path = new_model_path
