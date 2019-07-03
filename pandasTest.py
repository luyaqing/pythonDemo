# encoding=utf-8

import numpy as np
import pandas as pd
from pylab import *


def main():
    # Data Structure
    s = pd.Series([i*2 for i in range(1, 11)])
    print((type(s)))
    dates = pd.date_range("20170301", periods=8)
    df = pd.DataFrame(np.random.rand(8, 5), index=dates, columns=list("ABCDE"))     # index 表示 主键或者索引
    print(df)
    # df = pd.DataFrame({"A": 1, "B":pd.Timestamp("20170301"), "C": pd.Series(1, index=list(range(4)), dtype="float"),
    #                    "D": np.array([3]*4, dtype="float32"), "E": pd.Categorical(["police", "student", "teacher",
    #                                                                                "doctor"])})
    # print(df)

    # Basic
    print(df.head(3))               # 头三行
    print(df.tail(3))               # 最后三行
    print(df.index)                 # 主键
    print(df.values)                # 值
    print(df.T)                     # 转置
    print(df.sort_values("C"))     # C列排序  新版本的写法
    print(df.sort_index(axis=1, ascending=False))   # 第二个属性排序， 降序
    print(df.describe())                            # 数据的常规特征

    # Select
    print(df["A"])                  # 筛选列
    print(df[:3])                   # 利用数组筛选
    print(df["20170301":"20170304"])    # 利用日期筛选
    print(df.loc[dates[0]])             # 根据主键查询
    print(df.loc["20170301":"20170304", ["B", "D"]])    # 多维度查询
    print(df.at[dates[0], "C"])                         # 条件查询

    print(df.iloc[1:3, 2:4])                            # 条件查询
    print(df.iloc[1, 4])
    print(df.iat[1, 4])
    print(df[df.B > 0][df.A < 0])                       # 类似双边查询
    print(df[df > 0])
    print(df[df["E"].isin([1, 2])])                     # 在1,2 里面选择

    # set
    s1 = pd.Series(list(range(10, 18)), index=pd.date_range("20170301", periods=8))
    df["F"] = s1
    print(df)
    df.at[dates[0], "A"] = 0
    print(df)
    df.iat[1, 1] = 1
    df.loc[:, "D"] = np.array([4]*len(df))
    print(df)
    df2 = df.copy()
    df2[df2 > 0] = -df2
    print(df2)

    # Missing Values 缺失值的处理
    df1 = df.reindex(index=dates[:4], columns=list("ABCD")+["G"])
    df1.loc[dates[0]:dates[1], "G"] = 1
    print(df1)
    print(df1.dropna())                         # 丢失掉 缺失值
    print(df1.fillna(value=1))                  # 填充缺失值   value是填充的值  也可以用插值算法  进行插值

    # Statistic 统计
    print(df.mean())                    # 均值
    print(df.var())                     # 方差
    s = pd.Series([1, 2, 4, np.nan, 5, 7, 9, 10], index=dates)
    print(s)
    print(s.shift(2))                   # 往下移位
    print(s.diff())                     # 下面一个减去上面一个
    print(s.value_counts())             # 每个值出现的次数
    print(df.apply(np.cumsum))          # 累加
    print(df.apply(lambda x: x.max()-x.min()))          # 最大值减去最小值 就是 极差

    # Concat
    pieces = [df[:3], df[-3:]]
    print(pd.concat(pieces))
    left = pd.DataFrame({"key": ["X", "Y"], "value": [1, 2]})
    right = pd.DataFrame({"key": ["X", "Z"], "value": [3, 4]})
    print("LEFT:", left)
    print("RIGHT:", right)
    print(pd.merge(left, right, on="key", how="outer"))             # 相当于 sql 中join；how可以是 left  默认是inner
    df3 = pd.DataFrame({"A": ["a", "b", "c", "b"], "B": list(range(4))})
    print(df3.groupby("A").sum())                                   # 聚合函数  相同的主键加在一起

    # Reshape 交叉表
    import datetime
    df4 = pd.DataFrame({'A': ['one', "one", 'two', 'three'] * 6,
                        'B': ['a', 'b', 'c'] * 8,
                        'c': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
                        'D': np.random.randn(24),
                        'E': np.random.randn(24),
                        'F': [datetime.datetime(2017, i, 1) for i in range(1, 13)] +
                             [datetime.datetime(2017, i, 15) for i in range(1, 13)]})
    print(df4)
    print(pd.pivot_table(df4, values="D", index=["A", "B"], columns=["c"]))                 # 透视表
    # Time Series  时间序列
    t_exam = pd.date_range("20170301", periods=10, freq="S")
    print(t_exam)

    # Graph 绘图
    ts = pd.Series(np.random.rand(1000), index=pd.date_range("20140301", periods=1000))
    ts = ts.cumsum()
    ts.plot()
    show()

    # File
    df6 = pd.read_csv("./data/test.csv")
    print(df6)
    df7 = pd.read_excel("./data/test.xlsx")
    print(df7)
    df6.to_csv("./data/test2.csv")
    df7.to_excel("./data/test2.xlsx")


if __name__ == '__main__':
    main()
