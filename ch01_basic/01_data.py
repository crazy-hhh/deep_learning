import os
import pandas as pd
import torch

# os.makedirs("./data", exist_ok=True)
# data_file = os.path.join("./data","manual_data.csv")
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir,"..","data","manual_data.csv")
with open(data_file,'w') as f:
    f.write("NumRooms,Area,Price\n")
    f.write("3,1500,300000\n")
    f.write("4,2000,400000\n")
    f.write("2,NA,200000\n")
    f.write("NA,NA,250000\n")
    f.write("5,2500,500000\n")
data = pd.read_csv(data_file)
#print(data)
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
#print(inputs)

x,y = torch.tensor(inputs.values, dtype=torch.float32),torch.tensor(outputs.values, dtype=torch.float32)
print(x,'\n',y)#数值类型转换为张量格式
