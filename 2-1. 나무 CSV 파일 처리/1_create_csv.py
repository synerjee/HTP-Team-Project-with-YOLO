# 이 코드는 훈련 데이터에 상응하는 빈 CSV 파일을 생성하는 코드입니다.
# 검증 데이터의 경우, 빈 CSV 파일이 주어지므로 이 코드가 필요하지 않습니다.

import pandas as pd
import os

# 데이터셋마다 해당 사항이 다를수 있으니,
# 해당 파이썬 파일을 돌리기 전에 수정바랍니다.
data_name = 'tree'
csv_file_name = 'test_tree.csv'
txt_file_path = './runs/detect/predict/labels'
columns = ['id','branch_yn','root_yn','crown_yn','fruit_yn','gnarl_yn','loc','size']

df = pd.DataFrame(columns=columns)
df = df.set_index('id')

# 텍스트 파일을 확인하면서 CSV 파일을 채워갑니다.
for filename in os.listdir(txt_file_path):
    if filename.endswith('.txt'):
        with open(os.path.join(txt_file_path, filename)) as txt_file:

            # 파일 이름을 CSV ID로 변환합니다.
            line_id = filename.split(data_name)[0] + data_name

            df.loc[line_id] = ''

df.to_csv(csv_file_name)