import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib import font_manager as fm

# 设置绘图字体
font_path = r'C:\Windows\Fonts\STKAITI.TTF'
stheiti_font = fm.FontProperties(fname=font_path)

# 数据预处理
def load_data(file_path, sample_size=1202619):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                index, source_movie, movie_reco = parts
                data.append((index.strip(), source_movie.strip(), movie_reco.strip()))
    df = pd.DataFrame(data, columns=['Index', 'Source_Movie', 'Movie_Reco'])
    if df.empty:
        raise ValueError("数据文件是空。")
    initial_size = df.shape[0]
    df = df.replace('', np.nan)
    df = df.dropna(subset=['Source_Movie', 'Movie_Reco'])
    final_size = df.shape[0]
    print(f"初始行数: {initial_size}, 删除缺失值后行数: {final_size}")
    if final_size > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"随机选取 {sample_size} 条数据用于后续分析。")
    return df


# 构建稀疏共现矩阵
def build_movie_matrix(df):
    movies = pd.concat([df['Source_Movie'], df['Movie_Reco']]).unique()
    movie_to_id = {movie: idx for idx, movie in enumerate(movies)}
    id_to_movie = {idx: movie for movie, idx in movie_to_id.items()}

    rows, cols, data = [], [], []

    for _, row in df.iterrows():
        source_id = movie_to_id[row['Source_Movie']]
        reco_id = movie_to_id[row['Movie_Reco']]
        rows.append(source_id)
        cols.append(reco_id)
        data.append(1)

    n_movies = len(movies)
    movie_matrix = csr_matrix((data, (rows, cols)), shape=(n_movies, n_movies))

    return movie_matrix, movie_to_id, id_to_movie


# 计算相似度
def compute_similarity(movie_matrix):
    similarity_matrix = cosine_similarity(movie_matrix, dense_output=False)
    return similarity_matrix


# 推荐系统实现
def recommend_movies(movie_id, movie_to_id, id_to_movie, similarity_matrix, top_n=5):
    if movie_id not in movie_to_id:
        raise ValueError(f"电影 {movie_id} 不在数据中。")

    movie_idx = movie_to_id[movie_id]
    similar_movies = list(enumerate(similarity_matrix[movie_idx].toarray().flatten()))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    recommendations = [id_to_movie[idx] for idx, sim in similar_movies[1:top_n + 1] if sim > 0]
    return recommendations


# 评估推荐系统
def evaluate_recommendation(df, similarity_matrix, movie_to_id, id_to_movie, top_n=5):
    precision_list = []
    recall_list = []
    movie_performance = {}
    ground_truth = defaultdict(set)
    for _, row in df.iterrows():
        ground_truth[row['Source_Movie']].add(row['Movie_Reco'])
    for movie in ground_truth.keys():
        if movie not in movie_to_id:
            continue
        recommendations = recommend_movies(movie, movie_to_id, id_to_movie, similarity_matrix, top_n)
        recommended_set = set(recommendations)
        true_likes = ground_truth[movie]
        if recommended_set:
            precision = len(recommended_set & true_likes) / len(recommended_set)
        else:
            precision = 0
        recall = len(recommended_set & true_likes) / len(true_likes)
        precision_list.append(precision)
        recall_list.append(recall)
        movie_performance[movie] = {"precision": precision, "recall": recall}

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    return avg_precision, avg_recall, movie_performance


# 可视化
def visualize_similarity_matrix(similarity_matrix, id_to_movie, sample_size=10):
    sample_indices = np.random.choice(len(id_to_movie), size=sample_size, replace=False)
    sample_matrix = similarity_matrix[np.ix_(sample_indices, sample_indices)].toarray()
    sample_titles = [id_to_movie[idx] for idx in sample_indices]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_matrix, xticklabels=sample_titles, yticklabels=sample_titles, cmap='coolwarm', annot=True,
                fmt=".2f")

    plt.title("电影相似度矩阵（样本）", fontproperties=stheiti_font)
    plt.xlabel("推荐电影", fontproperties=stheiti_font)
    plt.ylabel("源电影", fontproperties=stheiti_font)

    plt.xticks(fontproperties=stheiti_font, rotation=45, ha='right')
    plt.yticks(fontproperties=stheiti_font, rotation=0)
    plt.show()


# 主函数
def main():
    file_path = '../data/connection.txt'
    df = load_data(file_path)

    if df.isnull().sum().sum() > 0:
        raise ValueError("数据中仍存在缺失值，请检查输入数据。")

    movie_matrix, movie_to_id, id_to_movie = build_movie_matrix(df)
    similarity_matrix = compute_similarity(movie_matrix)

    avg_precision, avg_recall, movie_performance = evaluate_recommendation(
        df, similarity_matrix, movie_to_id, id_to_movie, top_n=5
    )
    print(f"推荐系统评估 - 平均精准率: {avg_precision:.2f}, 平均召回率: {avg_recall:.2f}")

    high_performance_movies = sorted(movie_performance.items(), key=lambda x: (x[1]['precision'], x[1]['recall']), reverse=True)
    best_movie = high_performance_movies[0][0]
    print(f"效果最佳电影ID: {best_movie}, 精准率: {movie_performance[best_movie]['precision']:.2f}, 召回率: {movie_performance[best_movie]['recall']:.2f}")

    recommendations = recommend_movies(best_movie, movie_to_id, id_to_movie, similarity_matrix, top_n=5)
    print(f"推荐的电影: {recommendations}")

    visualize_similarity_matrix(similarity_matrix, id_to_movie, sample_size=10)


if __name__ == "__main__":
    main()
