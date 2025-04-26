
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
import pymysql
import statsmodels.api as sm
import numpy as np

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'tushare',
    'port': 3306,
    'charset': 'utf8'  # 字符集设置
}
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset={db_config['charset']}"
)
conn = pymysql.connect(**db_config)
chunk_size = 10000

# 获取华夏银行日线数据
df = pd.read_sql_query(
    """
    SELECT d.*, m.buy_lg_vol, m.sell_lg_vol, m.buy_elg_vol, m.sell_elg_vol, m.net_mf_vol, i.vol as i_vol, i.closes as i_closes
    FROM date_1 d JOIN moneyflows m on d.ts_code = m.ts_code and d.trade_date = m.trade_date
    LEFT JOIN index_daily i on i.trade_date = d.trade_date and i.ts_code = '399001.SZ'
    WHERE d.trade_date BETWEEN '2023-01-01' and '2024-01-01' and d.ts_code = '002229.SZ'
    """,
    conn,
    chunksize=chunk_size
)
df1 = pd.concat(df, ignore_index=True)
df1.head()
# 增加一列，股票的涨跌幅
df1['zd_closes'] = round(((df1['closes'] - df1['closes'].shift(1)) / df1['closes'].shift(1)), 2)
df1['zs_closes'] = round(((df1['i_closes'] - df1['i_closes'].shift(1)) / df1['i_closes'].shift(1)), 2)
df1['zs_vol'] = round(((df1['i_vol'] - df1['i_vol'].shift(1)) / df1['i_vol'].shift(1)), 2)
# 处理缺失值
df1 = df1.dropna(subset=['zd_closes', 'zs_closes', 'zs_vol']).reset_index(drop=True)
numeric_cols = df1.select_dtypes(include=['number']).columns.tolist()

# 选择需要进行主成分分析的自变量
X = df1[['vol', 'net_mf_vol', 'i_closes', 'zs_closes', 'buy_lg_vol','sell_lg_vol', 'buy_elg_vol','sell_elg_vol']]

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(np.cov(X, rowvar=False))
print('累计贡献率为:', round(eigenvalues[:5].sum()/eigenvalues.sum(), 4)*100, '%')
# 选择要保留的主成分个数
n_components = 5
top_eigenvectors = eigenvectors[:, :n_components]

# 计算主成分
principal_components = np.dot(X, top_eigenvectors)

# 将主成分添加到原数据中
data_pca = pd.concat([df1, pd.DataFrame(principal_components)], axis=1)
data_pca.columns = [*df1.columns, *[f'PC{i+1}' for i in range(n_components)]]

# 添加常数列前确保数据类型正确
X_pca = data_pca[[f'PC{i+1}' for i in range(n_components)]].copy()
X_pca = sm.add_constant(X_pca)

# 确保y的索引与X_pca一致
y = df1['zd_closes'].copy()

# 构建回归模型
model_pca = sm.OLS(y, X_pca)
# 拟合模型
result_pca = model_pca.fit()
# 输出结果
print("\n回归模型结果:")
print(result_pca.summary())

# 选择PC2，PC3，PC4作为新的自变量
X_pca_selected = data_pca[['PC2', 'PC3', 'PC4']]
X_pca_selected.columns = ['PC2', 'PC3', 'PC4']
# 添加常数列
X_pca_selected = sm.add_constant(X_pca_selected)
# 因变量
y = df1['zd_closes'].copy()
# 构建回归模型
model_pca_selected = sm.OLS(y, X_pca_selected)
# 拟合模型
result_pca_selected = model_pca_selected.fit()
# 输出回归模型结果
print("\n回归模型结果:")
print(result_pca_selected.summary())

X_pca_selected = data_pca[['PC2', 'PC3', 'PC4']]
X_pca_selected.columns = ['PC2', 'PC3', 'PC4']
# 绘制散点图
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
for i, col in enumerate(X_pca_selected.columns):
    axes[i].scatter(X_pca_selected[col], y, s=50, alpha=0.7)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('(y)')
    axes[i].set_title(f'{col} ')
plt.tight_layout()
plt.show()

for k in range(0, 5):
    string_y = f'CP{k+1} = '
    i = eigenvectors[k]
    for j in range(len(i)):
        if i[j] > 0:
            string_y = string_y + f'+{round(i[j], 2)}*X_{j+1}'
        else:
            string_y = string_y + f'{round(i[j], 2)}*X_{j+1}'
    if k != 2 and k != 4:
        print(string_y)