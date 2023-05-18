from src.data_process import read_SVC_data, split_dataset
from src.eval import evaluation
from src.visualization import *
from src.elm import ELM, TELM

if __name__ == '__main__':
    data, target = read_SVC_data("C:\\Users\\29147\\Documents\\WeChat Files\\wxid_7301883019012\\FileStorage\\File\\2023-05\\原始数据.xlsx", index=True)
    train_data, train_gt, val_data, val_gt = split_dataset(data, target, 5)

    mae_elm = [[],[]]
    r2_elm = [[],[]]
    mae_telm = [[],[]]
    r2_telm = [[],[]]


    hidden_size_start = 10
    hidden_size_end = 100


    hidden_size_cur = hidden_size_start
    while hidden_size_cur <= hidden_size_end:
        mae_elm[0].append(hidden_size_cur)
        r2_elm[0].append(hidden_size_cur)
        mae_telm[0].append(hidden_size_cur)
        r2_telm[0].append(hidden_size_cur)


        # 创建elm模型对象
        model_elm = ELM(input_dim=4225, hidden_dim=hidden_size_cur, output_dim=1)
        model_telm = TELM(input_dim=4225, hidden_dim=hidden_size_cur, output_dim=1)
        # 训练
        model_elm.fit(train_data, train_gt)
        model_telm.fit(train_data, train_gt)

        # 评估
        pred_elm = model_elm(val_data)
        pred_telm = model_telm(val_data)

        mae, r2 = evaluation(pred_elm, val_gt)
        mae_elm[1].append(mae)
        r2_elm[1].append(r2)

        mae, r2 = evaluation(pred_telm, val_gt)
        mae_telm[1].append(mae)
        r2_telm[1].append(r2)
        
        # 设置间隔为200
        hidden_size_cur += 1

    # 画图
    plot_mae_r2(mae_elm, r2_elm, mae_telm, r2_telm)