import torch


# 定义R方的函数
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

# 定义RMSE的函数
def rmse(output, target):
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(output, target))

# 定义评估方法
def evaluation(pred, ground_truth):
    '''
    评估mae和r2
    params:
        pred:预测
        ground_truth:标签
    '''
    # 定义平均绝对误差的损失函数
    l1_loss = torch.nn.L1Loss()

    # 计算平均绝对误差
    mae = l1_loss(ground_truth, pred)
    # 计算R方
    r2 = rmse(pred, ground_truth)

    return mae, r2
