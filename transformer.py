import torch

import torch

# 1. 加载整个模型（假设它是一个完整的模型对象）
full_model = torch.load("dpo_iterative.pt",weights_only=False)  # 你的完整模型文件

# 2. 提取 state_dict（权重参数）
state_dict = full_model.state_dict()

# 3. 保存 state_dict 到新的文件
torch.save(state_dict, "iterative_dpo.pt")

