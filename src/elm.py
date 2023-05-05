import torch
import torch.nn as nn
import torch.nn.functional as F


class ELM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # default_args
        self.args = {
            'input_dim': 32,                   # 输入层维度为 32
            'output_dim': 24,                  # 输出层维度为 24
            'hidden_dim': 64,                   # 隐藏层维度为 8
            'activation_fun': 'sigmoid'        # 激活函数为 sigmoid
        }

        self.args.update(kwargs)              # 用 kwargs 更新默认参数

        self.activation_fun = getattr(F, self.args["activation_fun"])  # 获取指定的激活函数

        self.fc1 = nn.Linear(self.args['input_dim'], self.args['hidden_dim'])   # 定义一个全连接层
        self.fc2 = nn.Linear(self.args['hidden_dim'], self.args['output_dim'], bias=False) # 定义另一个全连接层，输出层不需要偏置

    def forward(self, x):
        with torch.no_grad():   # 不需要反向传播，因此使用 no_grad() 上下文管理器，以减少内存消耗
            x = self.activation_fun(self.fc1(x))   # 计算隐藏层的输出
            return self.fc2(x)                      # 计算输出层的输出
        
    def fit(self, data, ground_truth):
        # 给定 N 条 data 和对应的 ground_truth，更新 fc2 的参数
        
        with torch.no_grad():   # 不需要反向传播，同样使用 no_grad() 上下文管理器
            hidden_mat = self.activation_fun(self.fc1(data))  # 计算隐藏层的输出

            beta = torch.matmul(torch.pinverse(hidden_mat), ground_truth)  # 计算 H 的广义逆，求解 beta
            self.fc2.weight = nn.Parameter(beta.T)    # 将 beta 转置后，作为新的权重更新 fc2的参数


class TELM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # default_args
        self.args = {
            'input_dim': 32,                   # 输入层维度为 32
            'output_dim': 24,                  # 输出层维度为 24
            'hidden_dim': 64,                   # 隐藏层维度为 8
            # 'activation_fun': 'sigmoid'        # 激活函数为 sigmoid
        }

        self.args.update(kwargs)              # 用 kwargs 更新默认参数

        # self.activation_fun = getattr(F, self.args["activation_fun"])  # 获取指定的激活函数

        self.fc1 = nn.Linear(self.args['input_dim'], self.args['hidden_dim'])   # 输入层
        self.fc2 = nn.Linear(self.args['hidden_dim'], self.args['hidden_dim'])   # 中间层
        self.fc3 = nn.Linear(self.args['hidden_dim'], self.args['output_dim'], bias=False) # 输出层，不需要偏置

    def DSIG(self, x):
        '''activate fun'''
        return (1-torch.exp(-x))/(1+torch.exp(-x))

    def DSIG_reverse(self, x):
        # return torch.log(2/(1-x))
        return -torch.log((1-x+1e-5)/(1+x+1e-5))

    def forward(self, x):
        with torch.no_grad():   # 不需要反向传播，因此使用 no_grad() 上下文管理器，以减少内存消耗
            x = self.DSIG(self.fc1(x))
            # x = self.DSIG(self.fc2(x))
            return self.fc3(x)                      
        
    def fit(self, data, ground_truth):
        # 给定 N 条 data 和对应的 ground_truth，更新 fc2 的参数
        
        with torch.no_grad():   # 不需要反向传播，同样使用 no_grad() 上下文管理器
            b = data.shape[0]

            # 求beta
                                                        # batch, input_dim
            hidden_mat_1 = self.DSIG(self.fc1(data))    # batch, hidden_dim
            beta = torch.matmul(torch.pinverse(hidden_mat_1), ground_truth)  # 计算 H 的广义逆，求解 beta
            self.fc3.weight = nn.Parameter(beta.T)    # 将 beta 转置后，作为新的权重更新 fc3的参数

            # 求θ和bias
            hidden_cat_one = torch.pinverse(torch.cat((hidden_mat_1,torch.ones(b,self.args['hidden_dim'])), dim=1))
            reverse_target_mul_betaR = self.DSIG_reverse(torch.matmul(ground_truth, torch.pinverse(beta)))
            theta_and_bias = torch.matmul(hidden_cat_one, reverse_target_mul_betaR)
            theta = theta_and_bias[0:self.args['hidden_dim']]
            bias = theta_and_bias[self.args['hidden_dim']:].sum(dim=0)

            self.fc2.weight = nn.Parameter(theta.T)    # 将 theta 转置后，作为新的权重更新 fc3的参数
            self.fc2.bias = nn.Parameter(bias)