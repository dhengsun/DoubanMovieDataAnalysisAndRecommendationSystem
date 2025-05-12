import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 读取数据
data = pd.read_excel("../data/data.xlsx")

# 设置绘图字体
font_path = 'C:/Windows/Fonts/STKAITI.TTF'
stheiti_font = fm.FontProperties(fname=font_path)

# 绘制评分、评分人数、电影时长的箱线图
plt.figure(figsize=(15, 5))

# 评分箱线图
plt.subplot(1, 3, 1)
plt.boxplot(data['rate'].dropna(), patch_artist=True, boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red"))
plt.title("评分箱线图", fontproperties=stheiti_font, fontsize=16)
plt.ylabel("评分", fontproperties=stheiti_font, fontsize=14)

# 评分人数箱线图
plt.subplot(1, 3, 2)
plt.boxplot(data['rating_num'].dropna(), patch_artist=True, boxprops=dict(facecolor="lightgreen"), medianprops=dict(color="red"))
plt.title("评分人数箱线图", fontproperties=stheiti_font, fontsize=16)
plt.ylabel("评分人数", fontproperties=stheiti_font, fontsize=14)

# 电影时长箱线图
plt.subplot(1, 3, 3)
plt.boxplot(data['runtime'].dropna(), patch_artist=True, boxprops=dict(facecolor="lightpink"), medianprops=dict(color="red"))
plt.title("电影时长箱线图", fontproperties=stheiti_font, fontsize=16)
plt.ylabel("电影时长（分钟）", fontproperties=stheiti_font, fontsize=14)

# 调整布局并保存图像
plt.tight_layout()
plt.savefig("boxplots.png", bbox_inches='tight', dpi=300)
plt.show()


