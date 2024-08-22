import json
import math

def calculate_metrics(actual, predicted):
    n = len(actual)
    mse = sum((a - p) ** 2 for a, p in zip(actual, predicted) if p != -1) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(a - p) for a, p in zip(actual, predicted) if p != -1) / n
    return rmse, mae

with open('path/to/test.json', 'r') as f:
    data1 = json.load(f)

with open('path/to/your/result/file', 'r') as f:
    data2 = json.load(f)


print("Structure of data2:")
print(type(data2))
print(data2.keys() if isinstance(data2, dict) else "Not a dictionary")

actual_scores = [float(item['score']) for item in data1]


if isinstance(data2, dict):
    max_prob_ratings = [float(data2[key]['max_prob_rating']) for key in data2]
    expected_rating = [float(data2[key]['expected_rating']) for key in data2]
else:
    print("Unexpected structure of data2. Please check the file content.")
    exit()
    
rmse_max_prob, mae_max_prob = calculate_metrics(actual_scores, max_prob_ratings)

rmse_expected_prob, mae_expected_prob = calculate_metrics(actual_scores, expected_rating)

print(f"\nCompared with max_prob_rating:")
print(f"RMSE: {rmse_max_prob:.4f}")
print(f"MAE: {mae_max_prob:.4f}")

print(f"\nCompared with expected_rating:")
print(f"RMSE: {rmse_expected_prob:.4f}")
print(f"MAE: {mae_expected_prob:.4f}")
