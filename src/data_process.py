import torch
import pandas as pd
import torch.nn as nn

def read_SVC_data(file_path, index=False, norm=True):
    df = pd.read_excel(file_path, header=None)
    data = torch.tensor(df.iloc[2:].values.astype(float), dtype=torch.float32)
    target = torch.tensor(df.iloc[1].values.astype(float), dtype=torch.float32).unsqueeze(1)

    if index:
        avg_pool = nn.AvgPool1d(15,15,1)
        data = data.transpose(0,1)
        pooled_data = avg_pool(data.unsqueeze(1)).squeeze(1)
        NDI = (pooled_data.unsqueeze(1)-pooled_data.unsqueeze(2))/(pooled_data.unsqueeze(1)+pooled_data.unsqueeze(2)+1e-5)
        NRI = pooled_data.unsqueeze(1)/(pooled_data.unsqueeze(2)+pooled_data.unsqueeze(1)+1e-5)      # The division operation results in 
                                                                        # a wide range of RI values, +self in the denominator to 
                                                                        # stabilizes the model performance

        NDI = torch.tril(NDI)
        NRI = torch.triu(NRI)
        INDEX = NDI + NRI
        return INDEX.view(-1, 65 ** 2), target

    # min-max normalization
    data_min = torch.min(data, dim=0)[0]
    data_max = torch.max(data, dim=0)[0]
    data_norm = (data - data_min) / (data_max - data_min)
    if not norm:
        return data.transpose(0,1), target
    else:
        return data_norm.transpose(0,1), target



def split_dataset(data, target, n_splits):
    '''
    拆分n折，拆分后的数据集有相同分布(未完成)
    params:
        data:数据
        target:标签
        n_splits:分割的份数
    '''
    # 按照ground truth的大小对数据进行降序排序
    sorted_indices = torch.argsort(target, dim=0, descending=False).squeeze()
    data_sorted = data[sorted_indices]
    target_sorted = target[sorted_indices]

    # 分割成5个子数据集，每个数据集中的品位分布大致相同

    sub_data = []
    sub_ground_truth = []

    for i in range(n_splits):
        sub_ground_truth.append([])
        sub_data.append([])


    for idx in range(data_sorted.__len__()):
        sub_data[idx%n_splits].append(data_sorted[idx].unsqueeze(0))
        sub_ground_truth[idx%n_splits].append(target_sorted[idx])


    # 将数据和标签从list转回tensor
    for i in range(n_splits):
        sub_ground_truth[i] = torch.as_tensor(sub_ground_truth[i], dtype=torch.float32)
        sub_data[i] = torch.cat(sub_data[i], dim=0)

    # sub_ground_truth = torch.cat(sub_ground_truth, dim=0)
    # sub_data = torch.cat(sub_data, dim=0)

    # 姑且先用最后一个当验证集，不交叉验证了，写着麻烦
    train_data = torch.cat(sub_data[:n_splits-1])
    train_gt = torch.cat(sub_ground_truth[:n_splits-1]).unsqueeze(1)

    val_data = torch.cat(sub_data[n_splits-1:])
    val_gt = torch.cat(sub_ground_truth[n_splits-1:]).unsqueeze(1)

    return train_data, train_gt, val_data, val_gt