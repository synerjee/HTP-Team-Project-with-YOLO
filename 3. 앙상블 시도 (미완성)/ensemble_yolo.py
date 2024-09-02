import torch
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# YOLOv8, YOLOv9, YOLOv10 모델 로드 
model_v8 = YOLO("C:/Users/코넷/Desktop/test/runs/detect/train20/weights/best.pt")
model_v9 = YOLO("C:/Users/코넷/Desktop/test/runs/detect/train20/weights/best.pt")
model_v10 = YOLO("C:/Users/코넷/Desktop/DeepLearning_Class/runs/detect/train9/weights/best.pt")

def ensemble_detection(image_path, confidence_threshold=0.25):
    """if문을 활용한 앙상블 함수

    Args:
        image_path : 테스트할 이미지 폴더의 경로
        confidence_threshold : 신뢰도의 임계값. 기본값은 0.25

    Returns:
        모델이 이미지에서 detect한 결과
    """
    # YOLOv8로 먼저 검출
    results_v8 = model_v8(image_path)
    
    # 결과 디버그용 출력
    print(f"YOLOv8 detection: {len(results_v8[0].boxes)} objects detected")
    
    # 검출된 객체가 없거나 confidence가 낮은 경우
    if len(results_v8[0].boxes) == 0 or results_v8[0].boxes.conf.max().item() < confidence_threshold:
        # YOLOv9로 검출 시도
        results_v9 = model_v9(image_path)
        print(f"YOLOv9 detection: {len(results_v9[0].boxes)} objects detected")
        
        # YOLOv9에서도 검출되지 않거나 confidence가 낮은 경우
        if len(results_v9[0].boxes) == 0 or results_v9[0].boxes.conf.max().item() < confidence_threshold:
            # YOLOv10으로 최종 검출 시도
            results_v10 = model_v10(image_path)
            print(f"YOLOv10 detection: {len(results_v10[0].boxes)} objects detected")
            return results_v10
        else:
            return results_v9
    else:
        return results_v8

def process_folder(input_folder, output_folder, gt_folder):
    """폴더 내 이미지를 처리하는 함수

    Args:
        input_folder : 테스트할 이미지 폴더의 경로
        output_folder : 결과 이미지를 저장할 폴더의 경로
        gt_folder : 정답 label 파일들이 있는 파일의 경로(mAP를 계산하기 위함)

    Returns:
        예측결과
    """
    # output_folder가 없을시 생성
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 지원하는 image 확장자
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    # 예측값과 실제값을 저장할 딕셔너리 초기화(클래스별로 구분하여 저장)
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    # input_folder의 모든 파일에 대해 반복하는 반복문
    for filename in os.listdir(input_folder):
        # 파일이 지원하는 확장자인 경우에만 처리
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"detected_{filename}")
            
            # 이미지에 대해 앙상블 검출 수행
            results = ensemble_detection(input_path)
            
            # 결과 저장
            img = cv2.imread(input_path)
            for r in results:
                img = r.plot()  # 검출 결과를 이미지에 그림(바운딩 박스를 그림)
            cv2.imwrite(output_path, img)
            
            # 예측 결과 저장
            img_height, img_width = img.shape[:2]
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist() # 바운딩 박스 좌표
                    x1, y1, x2, y2 = xyxy
                    x1 /= img_width
                    x2 /= img_width
                    y1 /= img_height
                    y2 /= img_height
                    conf = box.conf.item()  # 신뢰도
                    cls = box.cls.item()    # 클래스
                    all_predictions[int(cls)].append([filename, conf, x1, y1, x2, y2])
            
            # Ground truth 데이터 로드
            gt_path = os.path.join(gt_folder, filename.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as f:
                    for line in f:
                        cls, x_center, y_center, w, h = map(float, line.strip().split())
                        x1 = x_center - w/2
                        y1 = y_center - h/2
                        x2 = x_center + w/2
                        y2 = y_center + h/2
                        all_ground_truths[int(cls)].append([filename, 1.0, x1, y1, x2, y2])
            
            print(f"Processed: {filename}")
    
    # mAP 계산
    mAP = calculate_mAP(all_predictions, all_ground_truths)
    print(f"mAP: {mAP:.4f}")
    
    return all_predictions

def calculate_precision_recall(predictions, ground_truths, iou_threshold):
    """정밀도와 재현율을 계산하는 함수

    Args:
        predictions : 예측값
        ground_truths : 실제값
        iou_threshold : iou의 임계값

    Returns:
        정밀도, 재현율
    """
    # 예측을 신뢰도 점수에 따라 내림차순으로 정렬
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    
    total_gt = len(ground_truths)   # 실제 Ground Truth 바운딩 박스의 총 개수
    detected_gt = set() # 이미 탐지된 gt의 인덱스 저장
    tp = np.zeros(len(predictions)) # True Positive 개수를 저장하는 배열
    fp = np.zeros(len(predictions)) # False Positive 개수를 저장하는 배열
    
    # 각 예측에 대해 실제 Ground Truth와 비교
    for i, pred in enumerate(predictions):
        pred_filename = pred[0]
        pred_bbox = pred[2:]
        
        matched_gt = False
        for gt_idx, gt in enumerate(ground_truths):
            gt_filename = gt[0]
            gt_bbox = gt[2:]
            
            # 예측과 실제 파일 이름이 일치하고, 해당 Ground Truth가 아직 detect 되지 않은 경우
            if pred_filename == gt_filename and gt_idx not in detected_gt:
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou >= iou_threshold:
                    tp[i] = 1   # True Positive로 간주
                    detected_gt.add(gt_idx) # 탐지된 gt 인덱스 추가
                    matched_gt = True
                    break
        
        if not matched_gt:
            fp[i] = 1   # 예측이 일치하지 않으면 FP로 간주
    
    # 누적 합 계산
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # 정밀도와 재현율 계산
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_gt
    
    # 시작점 (0,1) 추가
    precision = np.concatenate(([1], precision))
    recall = np.concatenate(([0], recall))
    
    return precision, recall

def calculate_average_precision(precision, recall):
    # 정밀도-재현율 곡선 아래 영역 계산 (AP)
    return np.trapz(precision, recall)

def calculate_mAP(predictions, ground_truths, iou_threshold=0.5):
    """각 클래스에 대해 ap를 계산하고 평균 내서 mAP를 계산하는 함수

    Args:
        predictions : 예측값
        ground_truths : 실제값
        iou_threshold : iou의 임계값

    Returns:
        mAP
    """
    APs = []
    for cls in ground_truths.keys():
        if cls not in predictions:
            print(f"Class {cls} not in predictions")
            continue
        
        # 클래스별로 정밀도와 재현율을 계산
        precision, recall = calculate_precision_recall(predictions[cls], ground_truths[cls], iou_threshold)
        # 클래스별 AP를 계산하여 리스트에 추가
        AP = calculate_average_precision(precision, recall)
        print(f"Class {cls} AP: {AP:.4f}")
        APs.append(AP)
    
    # 클래스별 AP의 평균을 구하여 mAP 계산
    mAP = np.mean(APs) if APs else 0
    return mAP

def calculate_iou(box1, box2):
    """두 바운딩 박스의 iou를 계산하는 함수

    Args:
        box1 : 첫 번째 바운딩 박스의 좌표
        box2 : 두 번째 바운딩 박스의 좌표

    Returns:
        iou : 두 박스 사이의 iou값
    """

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # 교집합 영역 계산
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    
    # 각 박스의 넓이 계산
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    # iou 계산
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# 실행
input_folder = 'C:/Users/코넷/Desktop/DeepLearning_Class/test/images'
output_folder = 'C:/Users/코넷/Desktop/DeepLearning_Class/test/results'
gt_folder = 'C:/Users/코넷/Desktop/DeepLearning_Class/test/labels' 

process_folder(input_folder, output_folder, gt_folder)
