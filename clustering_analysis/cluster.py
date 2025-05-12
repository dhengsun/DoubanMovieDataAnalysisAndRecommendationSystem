import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns

# 设置绘图字体
font_path = 'C:/Windows/Fonts/STKAITI.TTF'
stheiti_font = fm.FontProperties(fname=font_path)

# 读取数据
data = pd.read_excel('../data/data.xlsx')

# 删除无关的列
data_cleaned = data.drop(columns=['title', 'year'])

# 演员列（actor），采用词频编码
actor_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), max_features=10)
actor_encoded = actor_vectorizer.fit_transform(data_cleaned['actor'].fillna(''))
actor_df = pd.DataFrame(actor_encoded.toarray(), columns=[f"actor_{name}" for name in actor_vectorizer.get_feature_names_out()])
# 导演列（directors），采用词频编码
director_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), max_features=5)
director_encoded = director_vectorizer.fit_transform(data_cleaned['directors'].fillna(''))
director_df = pd.DataFrame(director_encoded.toarray(), columns=[f"director_{name}" for name in director_vectorizer.get_feature_names_out()])
# 编剧列（scriptwriter），采用词频编码
scriptwriter_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), max_features=5)
scriptwriter_encoded = scriptwriter_vectorizer.fit_transform(data_cleaned['scriptwriter'].fillna(''))
scriptwriter_df = pd.DataFrame(scriptwriter_encoded.toarray(), columns=[f"scriptwriter_{name}" for name in scriptwriter_vectorizer.get_feature_names_out()])
# 电影类型列（type），采用多热编码
type_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), max_features=10)
type_encoded = type_vectorizer.fit_transform(data_cleaned['type'].fillna(''))
type_df = pd.DataFrame(type_encoded.toarray(), columns=[f"type_{name}" for name in type_vectorizer.get_feature_names_out()])
# 语言列（language），采用多热编码
language_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), max_features=5)
language_encoded = language_vectorizer.fit_transform(data_cleaned['language'].fillna(''))
language_df = pd.DataFrame(language_encoded.toarray(), columns=[f"language_{name}" for name in language_vectorizer.get_feature_names_out()])
# 地区列（region），采用多热编码
region_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), max_features=5)
region_encoded = region_vectorizer.fit_transform(data_cleaned['region'].fillna(''))
region_df = pd.DataFrame(region_encoded.toarray(), columns=[f"region_{name}" for name in region_vectorizer.get_feature_names_out()])
# 合并编码后的数据
processed_data = pd.concat(
    [
        data_cleaned[['rate', 'rating_num', 'runtime']],
        actor_df,
        type_df,
        director_df,
        scriptwriter_df,
        language_df,
        region_df
    ],
    axis=1
)
scaler = MinMaxScaler()
processed_data[['rate', 'rating_num', 'runtime']] = scaler.fit_transform(processed_data[['rate', 'rating_num', 'runtime']])

data = processed_data

cluster_range = range(2, 11)
ch_scores = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)

    # 计算 CH 指数
    ch_score = calinski_harabasz_score(data, kmeans.labels_)
    ch_scores.append(ch_score)

plt.figure(figsize=(12, 6))
plt.plot(cluster_range, ch_scores, marker='o', color='blue', label='Calinski-Harabasz 指数')
plt.xlabel('聚类数 (k)', fontsize=12, fontproperties=stheiti_font)
plt.ylabel('Calinski-Harabasz 指数', fontsize=12, fontproperties=stheiti_font)
plt.title('聚类数选择 - Calinski-Harabasz 指数', fontsize=14, fontproperties=stheiti_font)
plt.xticks(cluster_range, fontproperties=stheiti_font)
plt.yticks(fontproperties=stheiti_font)
plt.grid(alpha=0.5)
plt.legend(prop=stheiti_font)
plt.savefig('Calinski-Harabasz.png', bbox_inches='tight', dpi=300)
plt.show()

best_k_ch = cluster_range[np.argmax(ch_scores)]
print(f"根据 Calinski-Harabasz 指数，最佳的聚类数为：{best_k_ch}")

k = best_k_ch
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
data['cluster'] = kmeans.labels_
popularity_labels = {
    0: '冷门',
    1: '受欢迎',
    2: '最受欢迎'
}
data['popularity'] = data['cluster'].map(popularity_labels)
tsne = TSNE(n_components=2, random_state=42)
data_2d_tsne = tsne.fit_transform(data.drop(['cluster', 'popularity'], axis=1))
data_tsne = pd.DataFrame(data_2d_tsne, columns=['TSNE1', 'TSNE2'])
data_tsne['popularity'] = data['popularity']
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2', hue='popularity', palette='Set2', data=data_tsne, s=80, alpha=0.8
)
plt.title('基于 TSNE 的聚类结果可视化', fontsize=14, fontproperties=stheiti_font)
plt.xlabel('TSNE 维度 1', fontsize=12, fontproperties=stheiti_font)
plt.ylabel('TSNE 维度 2', fontsize=12, fontproperties=stheiti_font)
legend = plt.legend(title='受欢迎程度', fontsize=10, title_fontsize=12)
for text in legend.texts:
    text.set_fontproperties(stheiti_font)
legend.get_title().set_fontproperties(stheiti_font)
plt.grid(alpha=0.3)
plt.savefig('scatterplot.png', bbox_inches='tight', dpi=300)
plt.show()
