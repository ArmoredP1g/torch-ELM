import torch


# 定义R方的函数
def r2_score(y_true, y_pred):
    # 计算y_true和y_pred的均值
    y_true_mean = torch.mean(y_true)
    y_pred_mean = torch.mean(y_pred)
    # 计算y_true和y_pred的方差
    y_true_var = torch.sum((y_true - y_true_mean) ** 2)
    y_pred_var = torch.sum((y_pred - y_pred_mean) ** 2)
    # 计算y_true和y_pred的协方差
    y_true_pred_cov = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    # 计算R方
    r2 = (y_true_pred_cov ** 2) / (y_true_var * y_pred_var)
    return r2

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
    r2 = r2_score(ground_truth, pred)

    return mae, r2
