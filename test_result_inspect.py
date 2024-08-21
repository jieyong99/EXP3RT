
import json
import math

def calculate_metrics(actual, predicted):
    n = len(actual)
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted) if p != -1) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted) if p != -1) / n
    return rmse, mae

# JSON 파일 읽기
with open('data/amazon-book/rating_bias/test.json', 'r') as f:
    data1 = json.load(f)

with open('amazon-book_test_helpful_r128_alpha32_seed42_result.json', 'r') as f:
    data2 = json.load(f)

# data2의 구조 확인
print("Structure of data2:")
print(type(data2))
print(data2.keys() if isinstance(data2, dict) else "Not a dictionary")

# 점수 추출
actual_scores = [float(item['score']) for item in data1]

# data2의 구조에 따라 적절히 처리
if isinstance(data2, dict):
    max_prob_ratings = [float(data2[key]['max_prob_rating']) for key in data2]
    expected_rating = [float(data2[key]['expected_rating']) for key in data2]
    # expected_rating_tau_01 = [float(data2[key]['expected_rating_tau_0.1']) for key in data2]
    # expected_rating_tau_001 = [float(data2[key]['expected_rating_tau_0.01']) for key in data2]
else:
    print("Unexpected structure of data2. Please check the file content.")
    exit()

# max_prob_rating과 비교
rmse_max_prob, mae_max_prob = calculate_metrics(actual_scores, max_prob_ratings)


rmse_expected_prob, mae_expected_prob = calculate_metrics(actual_scores, expected_rating)


# rmse_expected_tau01, mae_expected_tau01 = calculate_metrics(actual_scores, expected_rating_tau_01)


# rmse_expected_tau001, mae_expected_tau001 = calculate_metrics(actual_scores, expected_rating_tau_001)



print(f"\nCompared with max_prob_rating:")
print(f"RMSE: {rmse_max_prob:.4f}")
print(f"MAE: {mae_max_prob:.4f}")

print(f"\nCompared with expected_rating:")
print(f"RMSE: {rmse_expected_prob:.4f}")
print(f"MAE: {mae_expected_prob:.4f}")

# print(f"\nCompared with expected_rating_tau_0.1:")
# print(f"RMSE: {rmse_expected_tau01:.4f}")
# print(f"MAE: {mae_expected_tau01:.4f}")

# print(f"\nCompared with expected_rating_tau_0.01:")
# print(f"RMSE: {rmse_expected_tau001:.4f}")
# print(f"MAE: {mae_expected_tau001:.4f}")
