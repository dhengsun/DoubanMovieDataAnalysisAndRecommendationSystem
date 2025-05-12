import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from matplotlib import font_manager as fm

# 设置绘图字体
font_path = 'C:\Windows\Fonts\STKAITI.TTF'
stheiti_font = fm.FontProperties(fname=font_path)

# 数据读取
data = pd.read_excel('../data/data.xlsx')

# 删除无关列
data_cleaned = data.drop(columns=['title', 'year'])


# 连续变量的分位数划分
def categorize_by_quantiles(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return pd.cut(series, bins=[-np.inf, q1, q3, np.inf], labels=['低评分', '中评分', '高评分'])

# 对连续变量进行分类
data_cleaned['rate_category'] = categorize_by_quantiles(data_cleaned['rate'])
data_cleaned['rating_num_category'] = categorize_by_quantiles(data_cleaned['rating_num'])
data_cleaned['runtime_category'] = categorize_by_quantiles(data_cleaned['runtime'])

# 对连续变量的类别列进行独热编码
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
rating_num_encoded = onehot_encoder.fit_transform(data_cleaned[['rating_num_category']])
runtime_encoded = onehot_encoder.fit_transform(data_cleaned[['runtime_category']])
rating_num_df = pd.DataFrame(rating_num_encoded, columns=['rating_num_1', 'rating_num_2'])
runtime_df = pd.DataFrame(runtime_encoded, columns=['runtime_1', 'runtime_2'])

# 演员列（actor），采用词频编码
actor_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), max_features=10)
actor_encoded = actor_vectorizer.fit_transform(data_cleaned['actor'].fillna(''))
actor_df = pd.DataFrame(actor_encoded.toarray(),
                        columns=[f"actor_{name}" for name in actor_vectorizer.get_feature_names_out()])

# 电影类型列（type），采用多热编码
type_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), max_features=10)
type_encoded = type_vectorizer.fit_transform(data_cleaned['type'].fillna(''))
type_df = pd.DataFrame(type_encoded.toarray(),
                       columns=[f"type_{name}" for name in type_vectorizer.get_feature_names_out()])

# 导演列（directors），采用词频编码
director_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), max_features=5)
director_encoded = director_vectorizer.fit_transform(data_cleaned['directors'].fillna(''))
director_df = pd.DataFrame(director_encoded.toarray(),
                           columns=[f"director_{name}" for name in director_vectorizer.get_feature_names_out()])

# 编剧列（scriptwriter），采用词频编码
scriptwriter_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(';'), max_features=5)
scriptwriter_encoded = scriptwriter_vectorizer.fit_transform(data_cleaned['scriptwriter'].fillna(''))
scriptwriter_df = pd.DataFrame(scriptwriter_encoded.toarray(), columns=[f"scriptwriter_{name}" for name in
                                                                        scriptwriter_vectorizer.get_feature_names_out()])

# 语言列（language），采用多热编码
language_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), max_features=5)
language_encoded = language_vectorizer.fit_transform(data_cleaned['language'].fillna(''))
language_df = pd.DataFrame(language_encoded.toarray(),
                           columns=[f"language_{name}" for name in language_vectorizer.get_feature_names_out()])

# 地区列（region），采用多热编码
region_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('/'), max_features=5)
region_encoded = region_vectorizer.fit_transform(data_cleaned['region'].fillna(''))
region_df = pd.DataFrame(region_encoded.toarray(),
                         columns=[f"region_{name}" for name in region_vectorizer.get_feature_names_out()])

# 合并所有特征
X = pd.concat(
    [
        actor_df,
        type_df,
        director_df,
        scriptwriter_df,
        language_df,
        region_df,
        rating_num_df,
        runtime_df
    ],
    axis=1
)

# 目标变量
y = data_cleaned['rate_category']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 贝叶斯优化范围
pbounds = {
    'n_estimators': (50, 200),
    'max_depth': (5, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}
# 存储迭代的指标
iterations = []
accuracies = []
f1_scores = {
    '低评分': [],
    '中评分': [],
    '高评分': []
}
def rf_objective_with_f1_metrics(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 计算总体准确率
    acc = accuracy_score(y_test, y_pred)
    # 记录每个类别的 F1 分数
    f1_scores['低评分'].append(f1_score(y_test, y_pred, labels=['低评分'], average='macro'))
    f1_scores['中评分'].append(f1_score(y_test, y_pred, labels=['中评分'], average='macro'))
    f1_scores['高评分'].append(f1_score(y_test, y_pred, labels=['高评分'], average='macro'))
    # 记录总体指标
    iterations.append(len(iterations) + 1)
    accuracies.append(acc)
    return acc
# 替换优化函数
optimizer = BayesianOptimization(
    f=rf_objective_with_f1_metrics,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=5, n_iter=25)

# 绘制三个类别 F1 分数的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, f1_scores['低评分'], label='低评分 F1 分数', marker='o', color='red')
plt.plot(iterations, f1_scores['中评分'], label='中评分 F1 分数', marker='x', color='green')
plt.plot(iterations, f1_scores['高评分'], label='高评分 F1 分数', marker='s', color='blue')
plt.title("贝叶斯优化过程中各类别 F1 分数变化曲线", fontproperties=stheiti_font, fontsize=16)
plt.xlabel("迭代次数", fontproperties=stheiti_font, fontsize=14)
plt.ylabel("F1 分数", fontproperties=stheiti_font, fontsize=14)
plt.legend(prop=stheiti_font, fontsize=12)
plt.grid()
plt.savefig("f1_score_change_curve.png", bbox_inches='tight', dpi=300)
plt.show()

# 绘制准确率变化曲线
plt.figure(figsize=(10, 6))
plt.plot(iterations, accuracies, label='准确率 (Accuracy)', marker='o', color='blue')
plt.title("贝叶斯优化过程中准确率变化曲线", fontproperties=stheiti_font, fontsize=16)
plt.xlabel("迭代次数", fontproperties=stheiti_font, fontsize=14)
plt.ylabel("准确率", fontproperties=stheiti_font, fontsize=14)
plt.legend(prop=stheiti_font, fontsize=12)
plt.grid()
plt.savefig("accuracy_change_curve.png", bbox_inches='tight', dpi=300)
plt.show()



# 获得最优参数
best_params = optimizer.max['params']
best_params = {
    'n_estimators': int(best_params['n_estimators']),
    'max_depth': int(best_params['max_depth']),
    'min_samples_split': int(best_params['min_samples_split']),
    'min_samples_leaf': int(best_params['min_samples_leaf'])
}
print("最优参数：", best_params)

# 使用最优参数训练最终模型
final_model = RandomForestClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# 评估指标
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵：\n", cm)
print("分类报告：\n", classification_report(y_test, y_pred))

# 绘制 ROC 曲线
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
y_test_binary = label_binarize(y_test, classes=['低评分', '中评分', '高评分'])
y_pred_prob = final_model.predict_proba(X_test)
plt.figure(figsize=(10, 6))
for i, label in enumerate(['低评分', '中评分', '高评分']):
    fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线
plt.title("ROC 曲线", fontproperties=stheiti_font, fontsize=16)
plt.xlabel("假阳性率 (FPR)", fontproperties=stheiti_font, fontsize=14)
plt.ylabel("真阳性率 (TPR)", fontproperties=stheiti_font, fontsize=14)
plt.legend(prop=stheiti_font, fontsize=12)
plt.grid()
plt.savefig("roc_curve.png", bbox_inches='tight', dpi=300)
plt.show()

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 4))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("随机森林混淆矩阵", fontproperties=stheiti_font, fontsize=14)
plt.xlabel("预测值", fontproperties=stheiti_font, fontsize=12)
plt.ylabel("真实值", fontproperties=stheiti_font, fontsize=12)
ax.set_xticklabels(['低评分', '中评分', '高评分'], fontproperties=stheiti_font, fontsize=10)
ax.set_yticklabels(['低评分', '中评分', '高评分'], fontproperties=stheiti_font, fontsize=10, rotation=0)
plt.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.show()
