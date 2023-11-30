import os
import pandas as pd
import sys

# 第一个参数时运行的目录
if(len(sys.argv) < 4) :
    print("使用 python get_rate.py 运行目录 开始时间戳 结束时间戳")
    os._exit(1)

run_dir = sys.argv[1]
os.chdir(run_dir)

# 拓宽500ms以满足窗口条件
start_time = int(sys.argv[2]) - 500
end_time = int(sys.argv[3]) + 500

data_two = pd.read_csv("./cpu_two.csv", header=None)
data_two.columns = ["name", "rate", "timestamp"]
data_two = data_two.loc[(data_two["timestamp"] > start_time) & (data_two["timestamp"] < end_time)]
# df=df.loc[df['score']>80]

# print(data_two)
# print(data_two.T)
# print(data_two.describe())
names = data_two["name"].unique()
with open("./rate_result.txt", "w", encoding="UTF-8") as f:
    f.writelines("CPU使用率:\n") 
    for name in names:
        # print(name)
        data = data_two[data_two['name'] == name]
        # print(data["rate"].mean())
        f.writelines(str(name) + ": " + str(data["rate"].mean()) + "\n")
    
    f.write("GPU使用率: \n")
    data_gpu = pd.read_csv("./gpu_rate.csv", header=None)
    data_gpu.columns = ["rate", "timestamp"]
    # index = 0
    # for rate in data_gpu["rate"]:
    #     if(rate != "0%"):
    #         break
    #     index += 1
    # # print(index)
    # # print(data_gpu)
    # data_gpu = data_gpu[index:]
    data_gpu = data_gpu[(data_gpu["timestamp"] >= start_time) & (data_gpu["timestamp"] <= end_time)]
    data_gpu["rate"] = data_gpu["rate"].str.strip('%').astype(float)/100
    # print(data_gpu)
    # print(data_gpu["rate"].mean())
    f.write(str(data_gpu["rate"].mean() * 100) + "\n")