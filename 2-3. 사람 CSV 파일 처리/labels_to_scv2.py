import pandas as pd
import os

# 데이터셋마다 해당 사항이 다를 수 있으니,
# 해당 파이썬 파일을 돌리기 전에 수정 바랍니다.
data_name = 'person'
csv_file_name = 'test_person.csv'
txt_file_path = './runs/detect/predict/labels'
classes = ['arm', 'eye', 'leg', 'mouth', 'size']
classes_column_dict = {
    'arm': 'arm_yn',
    'eye': 'eye_yn',
    'leg': 'leg_yn',
    'mouth': 'mouth_yn'
}
# 사람 데이터의 경우 0.4로 변경
big_size = 0.4

# 빈 CSV 파일을 채워넣는 부분입니다.
# 대부분의 데이터는 중앙에 위치해있고, 크기도 중간이라서
# 중앙 위치, 중간 크기를 디폴트 값으로 사용합니다.
empty_csv = pd.read_csv(csv_file_name)
id_set = set(empty_csv['id'].to_list())
empty_csv = empty_csv.set_index('id')
for col in empty_csv.columns:
    if col == 'loc':
        empty_csv[col] = 'center'
    elif col == 'size':
        empty_csv[col] = 'middle'
    elif col != 'id':
        empty_csv[col] = 'n'

# 텍스트 파일을 확인하면서 CSV 파일을 채워갑니다.
for filename in os.listdir(txt_file_path):
    if filename.endswith('.txt'):
        with open(os.path.join(txt_file_path, filename)) as txt_file:

            # 파일 이름을 CSV ID로 변환합니다.
            line_id = filename.split(data_name)[0] + data_name

            # CSV 파일에 주어진 파일만 처리하도록 합니다.
            if line_id not in id_set:
                continue

            # 해당 파일에서 발견된 부분들을 저장하는 set입니다.
            columns_found = set()

            for line in txt_file:
                class_id, center_x, center_y, width, height = [float(x) for x in line.split(' ')]
                class_id = int(class_id)

                # 사람 객체이면 해당 객체의 데이터를 이용해
                # 크기와 위치를 구합니다.
                if classes[class_id] == 'size':
                    size_label = 'middle'
                    size_ratio = width * height

                    if size_ratio >= big_size:
                        size_label = 'big'
                    elif size_ratio <= 0.16:
                        size_label = 'small'

                    loc = 'center'
                    if center_x < 0.33:
                        loc = 'left'
                    elif center_x > 0.67:
                        loc = 'right'

                    empty_csv.at[line_id, 'size'] = size_label
                    empty_csv.at[line_id, 'loc'] = loc

                # 다른 부위는 발견된 즉시 'n'을 'y'로 바꿉니다.
                elif classes[class_id] in classes_column_dict:
                    col = classes_column_dict[classes[class_id]]
                    if col not in columns_found:
                        columns_found.add(col)
                        empty_csv.at[line_id, col] = 'y'

empty_csv.to_csv(csv_file_name)
