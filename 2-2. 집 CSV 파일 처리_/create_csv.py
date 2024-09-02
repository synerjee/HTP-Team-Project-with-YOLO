import pandas as pd
import os

# data.yaml에서 설정을 반영한 경로와 클래스 정보
data_name = 'house'
csv_file_name = 'test_house.csv'
txt_file_path = 'C:/Users/코넷/Desktop/DeepLearning_Class/test/labels'
columns = ['id', 'door_yn', 'loc', 'roof_yn', 'window_cnt', 'size']

# 클래스 ID와 클래스 이름 매핑 (data.yaml에 기반)
class_map = {
    0: 'door',
    1: 'house',
    2: 'roof',
    3: 'window'
}

def get_size_category(area_percentage):
    if area_percentage >= 60:
        return 'big'
    elif area_percentage >= 16:
        return 'middle'
    else:
        return 'small'

def get_position_category(x_center):
    if x_center < 1/3:
        return 'left'
    elif x_center < 2/3:
        return 'center'
    else:
        return 'right'

def get_window_count_description(window_count):
    if window_count == 0:
        return '0'
    elif window_count == 1 or window_count == 2:
        return '1 or 2'
    else:
        return 'more than 3'

# 빈 데이터프레임 생성
df = pd.DataFrame(columns=columns)

# 텍스트 파일을 확인하면서 CSV 파일을 채워갑니다.
rows = []  # 새로운 행을 저장할 리스트
for filename in os.listdir(txt_file_path):
    if filename.endswith('.txt'):
        with open(os.path.join(txt_file_path, filename)) as txt_file:
            lines = txt_file.readlines()

            # 파일 이름을 CSV ID로 변환합니다.
            line_id = filename.split('.')[0]  # 확장자 제거한 파일명 사용

            # 기본값 설정
            door_count = 0
            roof_count = 0
            window_count = 0
            size_category = 'small'
            position_category = 'center'

            # 각 라인을 분석하여 개수 세기 및 크기와 위치 판별
            for line in lines:
                elements = line.split()
                class_id = int(elements[0])
                x_center = float(elements[1])
                y_center = float(elements[2])
                width = float(elements[3])
                height = float(elements[4])

                # 객체가 차지하는 면적 비율 계산
                area_percentage = (width * height) * 100  # 면적을 백분율로 변환

                # 객체의 크기와 위치 판별
                current_size_category = get_size_category(area_percentage)
                if current_size_category == 'big':
                    size_category = 'big'
                elif current_size_category == 'middle' and size_category == 'small':
                    size_category = 'middle'

                current_position_category = get_position_category(x_center)
                if current_position_category != 'center':
                    position_category = current_position_category

                # 클래스 ID에 따라 카운트 증가
                if class_id == 0:
                    door_count += 1
                elif class_id == 2:
                    roof_count += 1
                elif class_id == 3:
                    window_count += 1

            # 행 데이터를 딕셔너리로 추가
            row = {
                'id': line_id,
                'size': size_category,  # 이미지 내 객체의 크기
                'loc': position_category,  # 이미지 내 객체의 위치
                'roof_yn': 'y' if roof_count > 0 else 'n',  # 지붕이 하나라도 있으면 'y'
                'window_cnt': get_window_count_description(window_count),  # 창문의 개수 설명
                'door_yn': 'y' if door_count > 0 else 'n'  # 문이 하나라도 있으면 'y'
            }
            rows.append(row)

# 리스트를 데이터프레임으로 변환
df = pd.DataFrame(rows, columns=columns)

# CSV 파일로 저장
df.to_csv(csv_file_name, index=False)

print("CSV file created successfully!")
