import matplotlib.pyplot as plt

# 定义画图函数
def plot_mae_r2(mae_elm, r2_elm, mae_telm, r2_telm):
  # 创建一个一行两列的画布
  fig, (ax1, ax2) = plt.subplots(1, 2)
  # 设置画布大小
  fig.set_size_inches(10, 5)
  # 设置左侧子图的标题和坐标轴标签
  ax1.set_title("MAE")
  ax1.set_xlabel("hidden size")
  ax1.set_ylabel("MAE")
  # 设置左侧子图的横坐标刻度间隔为1000
#   ax1.xaxis.set_ticks(range(0, max(mae_elm[0], mae_telm[0]) + 1000, 1000))
  # 在左侧子图上画两条线段，分别对应ELM和TELM的MAE
  ax1.plot(mae_elm[0], mae_elm[1], color="blue", label="ELM")
  ax1.plot(mae_telm[0], mae_telm[1], color="red", label="TELM")
  # 显示左侧子图的图例
  ax1.legend()
  # 设置右侧子图的标题和坐标轴标签
  ax2.set_title("R2")
  ax2.set_xlabel("hidden size")
  ax2.set_ylabel("R2")
  # 设置右侧子图的横坐标刻度间隔为1000
  # ax2.xaxis.set_ticks(range(0, max(r2_elm[0], r2_telm[0]) + 1000, 1000))
  # 在右侧子图上画两条线段，分别对应ELM和TELM的R2
  ax2.plot(r2_elm[0], r2_elm[1], color="blue", label="ELM")
  ax2.plot(r2_telm[0], r2_telm[1], color="red", label="TELM")
  # 显示右侧子图的图例
  ax2.legend()
  # 显示整个画布
  plt.show()