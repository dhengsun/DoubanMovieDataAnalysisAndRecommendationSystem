import pandas as pd
data = pd.read_csv('../data/douban_movie.csv')
# 无关列删除
data = data.drop(columns=['IMDb', 'cover', 'crawled_at', 'id', 'url'])
# 异常值（评分）
data = data[(data['rate'] >= 0) & (data['rate'] <= 10)]
# 提取年份（年份）
data['year'] = data['Date'].str[:4]
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data.drop(columns = ['Date'], inplace = True)
# 提取电影时长
data['runtime'] = data['runtime'].str.extract(r'(\d+)').astype(float)
# 删除缺失值
data.dropna(inplace=True)
data.to_excel('../data/data.xlsx', index=False)


