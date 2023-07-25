import torch
# 两个真实的标号
y = torch.tensor([0,2])
# 预测值
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])

print(-torch.log(torch.tensor([0.1000, 0.5000])))