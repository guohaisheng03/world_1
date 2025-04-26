import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pymysql

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'tushare',
    'port': 3306,
    'charset': 'utf8'  # 字符集设置
}

engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?{db_config['charset']}"
)
conn = pymysql.connect(**db_config)
chunk_size = 10000

df = pd.read_sql_query(
    """
    SELECT d.* FROM date_1 d WHERE d.trade_date BETWEEN '2023-01-01' and '2024-01-01' and d.ts_code = '000001.SZ'
    """,
    conn,
    chunksize=chunk_size
)

df1 = pd.concat(df, ignore_index=True)

# 添加下一个交易日涨跌额
df1['zd_closes'] = round(((df1['closes'] - df1['closes'].shift(1)) / df1['closes'].shift(1)), 2)
# 处理缺失数据
df1 = df1.dropna(subset=['zd_closes'])
# print(df1.head)

ex = ["id", "ts_code", "trade_date", "the_date", "opens", "high", "low", "closes", "pre_closes", "changes", "pct_chg", "amount"]
number = df1.select_dtypes(include=['number']).columns.to_list()
newList = [col for col in number if col not in ex]

# 回归分析
formuls = 'zd_closes ~'+'+'.join(newList)
res = smf.ols(formuls, data=df1).fit()

print(res.summary())