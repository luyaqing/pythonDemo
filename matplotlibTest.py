# encoding=utf-8

import numpy as np


def main():

    import matplotlib.pyplot as plt

    '''
    # line
    x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    c, s = np.cos(x), np.sin(x)
    plt.figure(1)
    plt.plot(x, c, color="blue", linewidth=1.0, linestyle="-", label="COS", alpha=0.5)
    plt.plot(x, s, "r*", label="SIN")
    plt.title("CON & SIN")
    ax = plt.gca()        # 坐标轴设置
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.xaxis.set_ticks_position("bottom")            # 设置 x，y轴的位置
    ax.yaxis.set_ticks_position("left")
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],                       # 设置X，Y轴的值
               [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    plt.yticks(np.linspace(-1, 1, 5, endpoint=True))
    for label in ax.get_xticklabels()+ax.get_yticklabels():                 # 字体设置方块
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor="red", edgecolor="black", alpha=0.2))
    plt.legend(loc="upper left")                                            # 设置图例
    plt.grid()                                                              # 设置网格
    # plt.axis([-1, 1, -0.5, 1])                                              # 显示范围
    plt.fill_between(x, np.abs(x) < 0.5, c, c > 0.5, color="green", alpha=.25)            # 填充范围
    t = 1
    plt.plot([t, t], [0, np.cos(t)], "y", linewidth=3, linestyle="--")                    # 加了条线  y表示颜色
    plt.annotate("cos(1)", xy=(t, np.cos(t)), xycoords="data", xytext=(+10, +30),         # 加了注释
                 textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))
    plt.show()
    '''

    # scatter  散点图
    fig = plt.figure()
    fig.add_subplot(3, 3, 1)
    n = 128
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)                        # 颜色
    # plt.axes([0.025, 0.025, 0.95, 0.95])        # 显示范围
    plt.scatter(X, Y, s=75, c=T, alpha=.5)      # 画散点的  s是size   color
    plt.xlim(-1.5, 1.5), plt.xticks([])
    plt.ylim(-1.5, 1.5), plt.yticks([])
    plt.axis()
    plt.title("scatter")
    plt.xlabel("x")
    plt.ylabel("y")

    # bar  柱状图
    fig.add_subplot(332)                         # 3行3列 第二个位置
    n = 10
    X = np.arange(n)
    Y1 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X/float(n)) * np.random.uniform(0.5, 1.0, n)

    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    for x, y in zip(X, Y1):
        plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')       # ha 水平位置  va  垂直位置
    for x, y in zip(X, Y2):
        plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')

    # pie  饼图
    fig.add_subplot(333)
    n = 20
    Z = np.ones(n)
    Z[-1] *= 2
    plt.pie(Z, explode=Z * 0.5, colors=['%f' % (i / float(n)) for i in range(n)],
            labels=['%.2f' % (i / float(n)) for i in range(n)])                     # explode表示大小
    plt.gca().set_aspect('equal')
    plt.xticks(([])), plt.yticks([])

    # polar
    fig.add_subplot(334, polar=True)
    n = 20
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / n)            # 大小
    radii = 10 * np.random.rand(n)                              # 半径
    # plt.polar(theta, radii)
    plt.plot(theta, radii)

    # beatmap
    fig.add_subplot(335)
    from matplotlib import cm                                   # 上色用的
    data = np.random.rand(3, 3)
    cmap = cm.Blues
    map = plt.imshow(data, interpolation='nearest', cmap=cmap,  # interpolation 插值算法 aspect表示缩放，vmin是cmap大小 颜色
                     aspect='auto', vmin=0, vmax=1)

    # 3d
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(336, projection="3d")
    ax.scatter(1, 1, 3, s=100)

    # hot map
    fig.add_subplot(313)                                    # 3行1列第三个

    def f(x, y):
            return (1 - x / 2 + x ** 5 + Y ** 3) * np.exp(-x ** 2 - y ** 2)
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    plt.savefig("./data/fig.png")
    plt.show()


if __name__ == '__main__':
    main()
