import pandas as pd
import os

# 데이터셋마다 해당 사항이 다를수 있으니,
# 해당 파이썬 파일을 돌리기 전에 수정바랍니다.
data_name = 'tree'
submitted_csv = 'test_tree.csv'
answer_csv = 'train_tree.csv'
columns = ['branch_yn','root_yn','crown_yn','fruit_yn','gnarl_yn','loc','size']

# 제출 CSV와 정답 CSV를 판다스 데이터프레임으로 변환합니다.
submitted_df = pd.read_csv(submitted_csv)
submitted_df = submitted_df.set_index('id')
answer_df = pd.read_csv(answer_csv)
answer_df = answer_df.set_index('id')

correct = 0
total = len(columns) * len(submitted_df.index)

for id in submitted_df.index:
    submitted_row = submitted_df.loc[id]
    answer_row = answer_df.loc[id]

    # 맞춘 값마다 1점을 줍니다.
    for column in columns:
        if (submitted_row[column] == answer_row[column]):
            correct += 1

# 얼마나 맞췄는지 확인합니다
print(f"Correct: {correct}")
print(f"Total: {total}")
print(f"Score: {correct / total: .4f}")