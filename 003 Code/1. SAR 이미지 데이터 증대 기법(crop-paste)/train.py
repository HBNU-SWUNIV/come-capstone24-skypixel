import argparse
import warnings
import random
import os
import glob
import copy
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.optim as optim
from collections import Counter
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import wandb
import torchvision.transforms.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.vgg import vgg16
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings(action='ignore')

# 명령줄 인자 처리
parser = argparse.ArgumentParser(description='Train a Fast R-CNN network with wandb logging.')
parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training (default: 1)')
parser.add_argument('--img_size', type=int, default=512, help='Image size (default: 700)')
parser.add_argument('--data_usage', type=float, default=1.0, help='Fraction of data to use for training (default: 1.0)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--dataset_path', type=str, default='./data', help='Dataset path (default: ./data)')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
parser.add_argument('--num_classes', type=int, default=1, help='Number of classes (default: 1)')
parser.add_argument('--project_name', type=str, default='DA_faster_rcnn', help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default='dataset_name', help='Wandb run name')
args = parser.parse_args()

# wandb 설정
wandb.init(project=args.project_name, name=args.wandb_run_name, config=args)

# 환경 설정
CFG = {
    'SEED': 42,
    'NUM_CLASS': args.num_classes,
    'IMG_SIZE': args.img_size,
    'EPOCHS': args.epochs,
    'LR': args.lr,
    'BATCH_SIZE': args.batch_size,
    'DATASET_PATH': args.dataset_path,
    'NUM_WORKERS': args.num_workers,
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['SEED'])

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch] 
    
    images = torch.stack(images, dim=0)  
    
    return images, targets

class CustomDataset(Dataset):
    def __init__(self, root, train=True, transforms=None, data_usage=1.0):
        self.root = root
        self.train = train
        self.transforms = transforms
        self.data_usage = data_usage
        self.imgs = sorted(glob.glob(root + '/*.png'))
        self.imgs = self.imgs[:int(len(self.imgs) * data_usage)]  

        if train:
            self.boxes = sorted(glob.glob(root + '/*.txt'))
            self.boxes = self.boxes[:int(len(self.boxes) * data_usage)]

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        # 바운딩 박스 정보가 없는 경우, 빈 텐서 반환 대신 예외 처리
        if not lines:
            # 이 경우, 학습 과정에서 해당 데이터를 건너뛰거나, 적절한 처리가 필요함
            print(f"No bounding box information in {box_path}")
            return None, None

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min, x_max, y_max = values[1:5]

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img /= 255.0
        height, width = img.shape[0], img.shape[1]

        # 'image_id'를 추가하기 위한 파일 이름 또는 인덱스 사용
        image_id = os.path.basename(img_path).split('.')[0]

        if self.train:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1 # Background = 0

            if self.transforms is not None:
                transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
                img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
                
            # 여기에 'image_id' 추가
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])  # 또는 다른 고유 식별자
            }
            return img, target

        else:
            if self.transforms is not None:
                transformed = self.transforms(image=img)
                img = transformed["image"]
            
            # 검증/테스트 모드에서도 필요에 따라 'image_id' 추가 가능
            return img, {'image_id': torch.tensor([idx])}, width, height

    def __len__(self):
        return len(self.imgs)

def get_train_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_test_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        ToTensorV2(),
    ])

# 데이터셋 및 데이터로더 준비
train_dataset = CustomDataset(os.path.join(CFG['DATASET_PATH'], 'train'), train=True, transforms=get_train_transforms(), data_usage=args.data_usage)
val_dataset = CustomDataset(os.path.join(CFG['DATASET_PATH'], 'test'), train=True, transforms=get_train_transforms(), data_usage=args.data_usage)

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)

def build_model(num_classes):
    # COCO 데이터셋으로 사전 학습된 Faster R-CNN 모델 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 분류기의 입력 특징 수 얻기
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 분류기를 새로운 데이터셋의 클래스 수에 맞게 교체
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def getUnionAreas(boxA, boxB, interArea=None):
    area_A = getArea(boxA)
    area_B = getArea(boxB)
    if interArea is None:
        interArea = getIntersectionArea(boxA, boxB)    
    return float(area_A + area_B - interArea)

def getIntersectionArea(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # intersection area
    return max(0, xB - xA + 1) * max(0, yB - yA + 1)

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2] or boxB[0] > boxA[2]:
        return False  # boxA and boxB do not overlap
    if boxA[3] < boxB[1] or boxA[1] > boxB[3]:
        return False  # boxA and boxB do not overlap
    return True

def iou(boxA, boxB):
    if boxesIntersect(boxA, boxB) is False:
        return 0
    interArea = getIntersectionArea(boxA, boxB)
    union = getUnionAreas(boxA, boxB, interArea=interArea)
    result = interArea / union
    return result

def AP(detections, groundtruths, classes, IOUThreshold=0.5):
    eps = 1e-6
    APs = {}
    class_precisions = {}
    class_recalls = {}
    for cls in classes:
        cls_detections = [d for d in detections if d[1] == cls]
        cls_groundtruths = [g for g in groundtruths if g[1] == cls]
        npos = len(cls_groundtruths)
        cls_detections.sort(key=lambda x: x[2], reverse=True)
        TP = np.zeros(len(cls_detections))
        FP = np.zeros(len(cls_detections))
        matched_gt = []

        for d_idx, detection in enumerate(cls_detections):
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, gt in enumerate(cls_groundtruths):
                if gt[0] == detection[0]:
                    iou_score = iou(detection[3], gt[2])
                    if iou_score > max_iou:
                        max_iou = iou_score
                        max_gt_idx = gt_idx

            if max_iou >= IOUThreshold:
                if max_gt_idx not in matched_gt:
                    TP[d_idx] = 1
                    matched_gt.append(max_gt_idx)
                else:
                    FP[d_idx] = 1
            else:
                FP[d_idx] = 1

        cum_TP = np.cumsum(TP)
        cum_FP = np.cumsum(FP)
        precisions = cum_TP / (cum_TP + cum_FP + eps)
        recalls = cum_TP / (npos + eps)

        ap = np.trapz(precisions, recalls)
        APs[cls] = ap
        class_precisions[cls] = precisions
        class_recalls[cls] = recalls

    return APs, class_precisions, class_recalls

def mAP(APs):
    """
    Calculate the mean Average Precision (mAP) from the APs of all classes.
    """
    return sum(APs.values()) / len(APs)

def calculate_mAP_at_different_iou_thresholds(detections, groundtruths, classes, iou_thresholds):
    APs_at_iou_thresholds = {iou: [] for iou in iou_thresholds}
    
    for iou_threshold in iou_thresholds:
        APs, _, _ = AP(detections, groundtruths, classes, IOUThreshold=iou_threshold)
        for cls in classes:
            APs_at_iou_thresholds[iou_threshold].append(APs.get(cls, 0))
    
    mAPs = {iou: np.mean(APs) for iou, APs in APs_at_iou_thresholds.items()}
    return mAPs

def calculate_overall_mAP(mAPs):
    overall_mAP = np.mean(list(mAPs.values()))
    return overall_mAP

def F1_score(class_precisions, class_recalls):
    """
    각 클래스별 F1 score를 계산하고, 전체 클래스의 평균 F1 score를 반환합니다.
    """
    f1_scores = {}
    for cls in class_precisions.keys():
        precision = np.mean(class_precisions[cls])
        recall = np.mean(class_recalls[cls])
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores[cls] = f1_score
    
    # 전체 클래스의 평균 F1 score 계산
    avg_f1_score = np.mean(list(f1_scores.values()))
    return avg_f1_score

def visualize_prediction(image, boxes, labels):
    """
    이미지에 예측된 바운딩 박스를 그립니다.
    """
    image = image.cpu().detach()
    # 이미지 정규화 해제
    image = (image * 255).type(torch.uint8)
    
    # 이미지에 바운딩 박스 그리기
    drawn_image = draw_bounding_boxes(image, boxes, labels=labels, width=2, colors="red")
    return drawn_image

def train_and_validate(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # Train phase
        model.train()
        total_train_loss = 0
        total_class_loss = 0
        total_regression_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 클래스 분류 손실과 박스 회귀 손실을 따로 계산
            class_loss = loss_dict['loss_classifier']
            regression_loss = loss_dict['loss_box_reg']

            total_train_loss += losses.item()
            total_class_loss += class_loss.item()
            total_regression_loss += regression_loss.item()

            losses.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_regression_loss = total_regression_loss / len(train_loader)

        # Validation phase
        model.eval()
        detections = []
        groundtruths = []
        val_images = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                outputs = model(images)
                for i, output in enumerate(outputs):
                    pred_boxes = output['boxes'].cpu().numpy()
                    pred_scores = output['scores'].cpu().numpy()
                    pred_labels = output['labels'].cpu().numpy()

                    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                        detections.append((targets[i]['image_id'].item(), label, score, box))

                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    gt_labels = targets[i]['labels'].cpu().numpy()

                    for box, label in zip(gt_boxes, gt_labels):
                        groundtruths.append((targets[i]['image_id'].item(), label, box))

        classes = set(range(1, CFG['NUM_CLASS'] + 1))

        for idx, (image, targets) in enumerate(val_loader.dataset):
            image_id = targets['image_id'].item()  # 이미지 ID를 가져옴
            if not isinstance(image, torch.Tensor):
                image = F.to_tensor(image)
            image = image.unsqueeze(0).to(device)  # 모델 입력 형태로 변환

            outputs = model(image)

            # detach()를 호출하여 gradient 계산에서 텐서를 분리
            pred_boxes = outputs[0]['boxes'].detach().cpu().numpy().astype(int)
            pred_scores = outputs[0]['scores'].detach().cpu().numpy()
            pred_labels = outputs[0]['labels'].detach().cpu().numpy()

            # 시각화
            vis_image = visualize_prediction(image.squeeze(0), torch.tensor(pred_boxes), [f"{label}: {score:.2f}" for label, score in zip(pred_labels, pred_scores)])
            val_images.append(wandb.Image(vis_image, caption=f"Validation Image {image_id}"))

        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        mAPs_at_different_iou = calculate_mAP_at_different_iou_thresholds(detections, groundtruths, classes, iou_thresholds)
        overall_mAP = calculate_overall_mAP(mAPs_at_different_iou)

        APs, class_precisions, class_recalls = AP(detections, groundtruths, classes)
        avg_f1_score = F1_score(class_precisions, class_recalls)
        mAP_0_5 = mAP(APs)

        APs, _, _ = AP(detections, groundtruths, classes, IOUThreshold=0.6)
        mAP_0_6 = mAP(APs)

        APs, _, _ = AP(detections, groundtruths, classes, IOUThreshold=0.7)
        mAP_0_7 = mAP(APs)

        avg_precision = np.mean([np.mean(prec) for prec in class_precisions.values()])
        avg_recall = np.mean([np.mean(rec) for rec in class_recalls.values()])

        # 에폭이 끝날 때 모든 지표와 사진을 한 번에 로깅
        wandb.log({
            'Epoch': epoch+1,
            'Train Loss': avg_train_loss,
            'Class Loss': avg_class_loss,
            'Regression Loss': avg_regression_loss,
            'mAP_0.5': mAP_0_5,
            'mAP_0.6': mAP_0_6,
            'mAP_0.7': mAP_0_7,
            'mAP_0.5_0.95': overall_mAP,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1_score': avg_f1_score,
            'Validation Images': val_images
        })

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_model_wts)
    return model


# 모델, 옵티마이저, 스케줄러 초기화
model = build_model(num_classes=CFG['NUM_CLASS'] + 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['LR'], weight_decay=0.1)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'], eta_min=0)

# 학습 및 검증 실행
trained_model = train_and_validate(model, train_loader, val_loader, optimizer, scheduler, device, CFG['EPOCHS'])

# 모델 저장
torch.save(trained_model.state_dict(), 'best_model.pth')