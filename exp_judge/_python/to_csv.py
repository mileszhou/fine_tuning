import json
from pathlib import Path
from io import StringIO

from numpy import sqrt, std

# ----------- CONFIG ------------- 
T= 0.0
input_dir = "./_results/judge"
input_fn = f"judgement_T={T}.jsonl"     # input file with judgement and scores from the training comparison step
input_path = Path(input_dir) / input_fn
results_dir = "./_results/judge"
result_fn = f"judgement_T={T}.csv"
result_path = Path(results_dir) / result_fn
# --------------------------------

fi = open(input_path, "r", encoding="utf-8")
fo = open(result_path, "w", encoding="utf-8")

fo.write("index,winner,confidence,accuracy_1,accuracy_2,completeness_1,completeness_2,safety_1,safety_2\n")  # CSV header

winning = 0
winning_square = 0
x_i_sum = {"accuracy": [0, 0], "completeness": [0, 0], "safety": [0, 0]}
dx_i_sum = {"accuracy": 0, "completeness": 0, "safety": 0}
dx_i_squared_sum = {"accuracy": 0, "completeness": 0, "safety": 0}

for line in fi:
    item = json.loads(line)
    idx = item["idx"]
    # Calculate sum
    tmp = (item['winner']-1.5)*2*item['confidence']         # Track winning score (positive means model 2 is winning, negative means model 1 is winning)
    winning += tmp
    winning_square += tmp*tmp

    # Write winner/confidence to CSV
    fo.write(f"{idx},")
    fo.write(f"{item['winner']},{item['confidence']}")

    for key in ["accuracy", "completeness", "safety"]:
        # Update sums for mean and variance calculation
        for i in range(2): 
            fo.write(f",{item['scores'][i][key]}")   
            x_i_sum[key][i] += item["scores"][i][key]
        
        # Calculate differernces for average and std_dev calculation
        tmp = item['scores'][1][key] - item['scores'][0][key]               # Difference
        dx_i_sum[key] += tmp
        dx_i_squared_sum[key] += tmp*tmp
    fo.write("\n")  # end of line

print(f"Processed index {idx}...")

# Write average line
winning /= (idx+1)  # Average winning score
print(f"Average winning score (positive means model 2 is winning, negative means model 1 is winning): {winning:.4f}")
fo.write(f"average,{winning:.2f},")   # Spaces: -index, winning, -confidence 

d_mean = {"accuracy": 0, "completeness": 0, "safety": 0}
std_dev = {"accuracy": 0, "completeness": 0, "safety": 0}
for key in ["accuracy", "completeness", "safety"]:
    for i in range(2): 
        x_i_sum[key][i] /= (idx+1)   # x_i_sum is now average
        print(f"  mean {key}_{i+1} = {x_i_sum[key][i]:.2f}")
        fo.write(f",{x_i_sum[key][i]:.2f}")

    d_mean[key] = dx_i_sum[key] / (idx+1)
    std_dev[key] = sqrt((dx_i_squared_sum[key] - (idx+1)*d_mean[key]**2)/idx)   # idx==(idx+1)-1
    print(f"  {key} benefit: {d_mean[key]:.2f} (std dev: {std_dev[key]:.2f} = {d_mean[key]/std_dev[key]:.4f} estimated sigma)")
fo.write("\n")

# Write std_dev line
fo.write(f"std_dev,{sqrt((winning_square-(idx+1)*winning*winning)/idx):.2f},")
for key in ["accuracy", "completeness", "safety"]:
    fo.write(f",{d_mean[key]:.2f},{std_dev[key]:.2f}")
fo.write("\n")

fi.close()
fo.close()
print(f"Done! {idx+1} results saved to {result_path}.\n")
