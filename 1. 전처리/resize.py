from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size):
    # 출력 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 입력 폴더의 모든 파일을 순회
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff')):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.resize(new_size, Image.LANCZOS)
                    img.save(os.path.join(output_folder, filename))
                    print(f"Resized and saved {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

# 입력 폴더와 출력 폴더 경로 설정
input_folder = 'C:/Users/CONET-07/Desktop/train_data/cropped_images'
output_folder = 'C:/Users/CONET-07/Desktop/train_data/1K'
new_size = (1024, 768)

# 이미지 리사이즈 실행
resize_images(input_folder, output_folder, new_size)
