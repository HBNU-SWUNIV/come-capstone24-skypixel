import os
import cv2
from random import choice, randint
from shutil import copyfile
from tqdm import tqdm
import argparse

def augment_dataset_with_crops(dataset_dir, crops_dir, num_augmentations=2):
    bins_dir = [crops_dir]  # 단일 폴더에서 랜덤하게 선택
    augmented_dir = os.path.join(dataset_dir, 'augmented')
    os.makedirs(augmented_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(dataset_dir), desc="Processing images"):
        if img_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(dataset_dir, img_file)
            image = cv2.imread(image_path)
            h, w, _ = image.shape
            label_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
            
            # 원본 이미지에서 객체 정보를 읽어옵니다.
            objects_info = []
            existing_boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    for line in file:
                        objects_info.append(line)
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, x_min, y_min, x_max, y_max = map(float, parts)
                            existing_boxes.append([x_min, y_min, x_max, y_max])

            # 9개의 증강된 이미지를 생성합니다.
            for i in range(num_augmentations):
                augmented_image = image.copy()
                augmented_objects_info = objects_info.copy()
                
                selected_crops = [os.path.join(bins_dir[0], choice(os.listdir(bins_dir[0]))) for _ in range(randint(1, 3))]
                for crop_path in selected_crops:
                    crop_img = cv2.imread(crop_path)
                    crop_h, crop_w, _ = crop_img.shape
                    
                    # 무작위 위치 선택 전에 크기 확인
                    for _ in range(5000):  # 충돌을 피하기 위해 최대 100번 시도
                        x_min = randint(0, w - crop_w)
                        y_min = randint(0, h - crop_h)
                        x_max = x_min + crop_w
                        y_max = y_min + crop_h
                        
                        # 기존 객체와 충돌하는지 확인
                        collision = False
                        for box in existing_boxes:
                            if not (x_max < box[0] or x_min > box[2] or y_max < box[1] or y_min > box[3]):
                                collision = True
                                break
                        
                        if not collision:
                            # 이미지에 객체 붙여넣기
                            augmented_image[y_min:y_max, x_min:x_max] = crop_img
                            # 레이블에 객체 정보 추가하기
                            augmented_objects_info.append(f'0 {x_min} {y_min} {x_max} {y_max}\n')
                            break

                # 증강된 이미지와 레이블 저장
                augmented_img_filename = f"{img_file.split('.')[0]}_augmented_{i}.{img_file.split('.')[1]}"
                augmented_image_path = os.path.join(augmented_dir, augmented_img_filename)
                cv2.imwrite(augmented_image_path, augmented_image)
                
                augmented_label_filename = f"{img_file.split('.')[0]}_augmented_{i}.txt"
                augmented_label_path = os.path.join(augmented_dir, augmented_label_filename)
                with open(augmented_label_path, 'w') as file:
                    file.writelines(augmented_objects_info)

    print(f"Dataset augmentation complete. Created {num_augmentations} augmented versions for each image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment dataset with crops.')
    parser.add_argument('dataset_dir', type=str, help='Path to the original dataset directory')
    parser.add_argument('crops_dir', type=str, help='Path to the crops directory')
    parser.add_argument('--num_augmentations', type=int, default=2, help='Number of augmentations per image')
    
    args = parser.parse_args()
    augment_dataset_with_crops(args.dataset_dir, args.crops_dir, args.num_augmentations)
