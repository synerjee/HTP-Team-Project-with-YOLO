from PIL import Image
import os

# 이미지를 불러올 디렉토리와 크롭한 이미지를 저장할 디렉토리 설정
input_directory = 'C:/Users/CONET-07/Desktop/train_data/data/person'
output_directory = 'cropped_images'
os.makedirs(output_directory, exist_ok=True)

# 크롭할 좌표 설정 (왼쪽, 위, 오른쪽, 아래)
crop_box = (0, 725, 6760, 4785)

# 디렉토리 내의 모든 이미지 파일을 불러와 크롭 후 저장
for filename in os.listdir(input_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        with Image.open(os.path.join(input_directory, filename)) as img:
            # 이미지 크롭
            img_cropped = img.crop(crop_box)
            # 크롭한 이미지 저장
            img_cropped.save(os.path.join(output_directory, filename))

print("이미지 크롭이 완료되었습니다.")