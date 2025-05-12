import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm
from wordcloud import WordCloud
from collections import Counter
from matplotlib import colormaps

# 设置绘图字体
font_path = 'C:/Windows/Fonts/STKAITI.TTF'
stheiti_font = fm.FontProperties(fname=font_path)

# 读取数据
data = pd.read_excel('../data/data.xlsx')

# 数据提取
data = data.dropna(subset=['year'])
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data = data.dropna(subset=['year'])

# 将年份分组为每5年一组
bins = list(range(1989, 2025, 5))
labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
data['year_group'] = pd.cut(data['year'], bins=bins, labels=labels, right=False)
# 统计每组电影数量
year_group_counts = data['year_group'].value_counts().sort_index()
year_group_counts = year_group_counts.reset_index()
year_group_counts.columns = ['year_group', 'counts']
# 绘制饼图
sns.set_palette("coolwarm", len(year_group_counts))  # 从 Seaborn 中选取 coolwarm 配色方案
colors = sns.color_palette("coolwarm", len(year_group_counts))
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    year_group_counts['counts'],
    labels=year_group_counts['year_group'],
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    pctdistance=0.8,
    textprops={'fontsize': 12, 'fontproperties': stheiti_font},
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    shadow=True
)
ax.set_title(
    '1989-2020 每五年电影数量分布',
    fontproperties=stheiti_font,
    fontsize=16,
    color='darkblue',
    weight='bold',
    pad=20
)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontproperties(stheiti_font)
    autotext.set_fontsize(10)
legend_labels = year_group_counts['year_group'].tolist()
legend_colors = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors]
legend = ax.legend(
    legend_colors,
    legend_labels,
    loc='upper left',
    bbox_to_anchor=(1.05, 1.0),
    title="年份分组",
    fontsize=12,
    title_fontsize=14,
    frameon=False
)
legend.set_title("年份分组", prop=stheiti_font)
for text in legend.texts:
    text.set_fontproperties(stheiti_font)
plt.savefig('year_pie.png', bbox_inches='tight', dpi=300)
plt.show()


# 统计词频
def get_word_frequencies(text_list, delimiter=';'):
    all_words = []
    for text in text_list:
        words = text.split(delimiter)
        all_words.extend([word.strip() for word in words if word.strip()])
    word_freq = Counter(all_words)
    return dict(word_freq)
# 生成词云
def generate_wordcloud_from_frequencies(word_freq, title, font_path, file_name):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path=font_path,
        colormap='viridis',
        max_words=200
    ).generate_from_frequencies(word_freq)

    # 绘制词云
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontproperties=stheiti_font, fontsize=16)
    plt.axis('off')

    # 保存图形
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.show()


# 演员词云
actor_freq = get_word_frequencies(data['actor'].tolist(), delimiter=';')
generate_wordcloud_from_frequencies(actor_freq, "演员词云", font_path, "actor_wordcloud.png")

# 导演词云
director_freq = get_word_frequencies(data['directors'].tolist(), delimiter=';')
generate_wordcloud_from_frequencies(director_freq, "导演词云", font_path, "director_wordcloud.png")

# 编剧词云
scriptwriter_freq = get_word_frequencies(data['scriptwriter'].tolist(), delimiter=';')
generate_wordcloud_from_frequencies(scriptwriter_freq, "编剧词云", font_path, "scriptwriter_wordcloud.png")

# 电影类型词云
type_freq = get_word_frequencies(data['type'].tolist(), delimiter='/')
generate_wordcloud_from_frequencies(type_freq, "电影类型词云", font_path, "type_wordcloud.png")


# 绘制频数分布图
def plot_frequency_distribution(word_freq, title, xlabel, ylabel, font_path, file_name, top_n=10, cmap_name='viridis'):

    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, counts = zip(*sorted_freq)  # 解包为词语和对应的频次

    cmap = colormaps[cmap_name]
    colors = cmap([i / top_n for i in range(top_n)])

    sns.set_theme(style="ticks")
    sns.color_palette("pastel")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(words, counts, color=colors, edgecolor="black", linewidth=0.6)
    plt.title(title, fontproperties=stheiti_font, fontsize=16, weight="bold")
    plt.xlabel(xlabel, fontproperties=stheiti_font, fontsize=12)
    plt.ylabel(ylabel, fontproperties=stheiti_font, fontsize=12)
    plt.xticks(fontproperties=stheiti_font, fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count),
                 ha='center', va='bottom', fontsize=10, fontproperties=stheiti_font)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.show()


# 语言频数分布图
language_freq = get_word_frequencies(data['language'].tolist(), delimiter='/')
plot_frequency_distribution(language_freq, "语言频数分布图", "语言", "频数", font_path, "language_frequency.png")

# 地区频数分布图
region_freq = get_word_frequencies(data['region'].tolist(), delimiter='/')
plot_frequency_distribution(region_freq, "地区频数分布图", "地区", "频数", font_path, "region_frequency.png")