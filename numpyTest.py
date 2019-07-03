# encoding=utf-8


import numpy as np
from numpy.linalg import *


def main():
    lst = [[1, 3, 5], [2, 4, 6]]
    print(type(lst))
    mp_lst = np.array(lst)
    print(type(mp_lst))
    np_lst = np.array(lst, dtype=np.float)
    print(np_lst)
    print(np_lst.shape)  # 指明了形状
    print(np_lst.ndim)  # 指明了维度
    print(np_lst.dtype)  # 指明了类型
    print(np_lst.itemsize)  # 指明了大小，即字节 8
    print(np_lst.size)  # 指明了元素数量  6个
    # 2 Some Arrays
    print(np.zeros([2, 4]))  # 全是0的矩阵
    print(np.ones([3, 5]))  # 全是1的矩阵
    print(np.random.rand(2, 4))  # 0-1之间的分布均匀的随机数
    print(np.random.rand())  # 打印一个随机数
    print(np.random.randint(1, 100, 3))  # 打印一个随机整数 连续生产3个  默认是1个
    print(np.random.randn(2, 4))  # 打印正太分布的随机数  可无参数  是一个
    print(np.random.choice([10, 20, 30]))  # 从里面选择一个
    print(np.random.beta(1, 10, 10))  # B分布的随机数
    # 3 Array Opes
    print(np.arange(1, 11).reshape([2, -1]))  # -1表示缺省 代表5；前面一个表示等差数列  后面一个表示构造出2X5的矩阵
    lst = np.arange(1, 11).reshape([2, -1])
    print(np.exp(lst))  # 自然指数
    print(np.exp2(lst))  # 指数的平方
    print(np.sqrt(lst))  # 开方
    print(np.sin(lst))  # 正弦
    print(np.log(lst))  # 对数
    lst = np.array([[[1, 2, 3, 4],
                     [4, 5, 6, 7]],
                    [[7, 8, 9, 10],
                     [10, 11, 12, 13]],
                    [[14, 15, 16, 17],
                     [18, 19, 20, 21]]
                    ])
    print("多维的lst")
    print(lst.shape)
    print(lst.sum())
    print(lst.sum(axis=1))  # axis 表示计算的深度
    print(lst.max(axis=1))  # 可不写参数
    print(lst.min(axis=0))  # 可不写参数

    lst1 = np.array([10, 20, 30, 40])
    lst2 = np.array([4, 3, 2, 1])
    print(lst1+lst2)    # 加法操作
    print(lst1-lst2)    # 减法操作
    print(lst1*lst2)    # 乘法操作
    print(lst1/lst2)    # 除法操作
    print(lst1**2)      # 乘方操作

    print(np.dot(lst1.reshape([2, 2]), lst2.reshape([2, 2])))  # 点乘 矩阵
    print(np.concatenate((lst1, lst2), axis=0))     # 两个数据的追加
    print(np.vstack((lst1, lst2)))      # 两个数据的上下的追加
    print(np.hstack((lst1, lst2)))      # 两个是数据的追加 左右
    print(np.split(lst1, 4))            # 数据的分离
    print(np.copy(lst1))                # 对数据的拷贝

    print(np.eye(3))                    # 单位矩阵
    lst = np.array([[1., 2.],
                    [3., 4.]])
    print(inv(lst))                     # 矩阵的逆运算
    print(lst.transpose())              # 转置矩阵
    print(det(lst))                     # 行列式
    print(eig(lst))                     # 特征值  第一个是特征值  第二个是特征向量
    y = np.array([[5.], [7.]])
    print(solve(lst, y))                # 求解方程
    print(np.fft.fft(np.array([1, 1, 1, 1, 1, 1, 1])))   # FFT计算
    print(np.corrcoef([1, 0, 1], [0, 2, 1]))             # #coef的系数
    print(np.poly1d([2, 1, 3]))                          # 多元函数


if __name__ == '__main__':
    main()
