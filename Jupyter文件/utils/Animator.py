class Animator:  #@save
    """在动画中绘制数据
       useage:
           animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
           after update:
           animator.add(epoch + 1, train_metrics + (test_acc,))
    """
    # x、y轴标签，legend图例、xlim设置轴的显示范围、scale刻度尺度
    # fmts绘图格式，线的格式 nrows子图行数、列数 figsize图形尺寸
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        # 图例为空则初始化为一个空列表
        if legend is None:
            legend = []
        # 设置图片显示为svg格式
        d2l.use_svg_display()
        # subplots指定创建的子图的行数 子图的列数 figsize指定图形尺寸
        # fig当中保存图形对象，控制整个图行，保存、修改标题、设置背景色
        # axes保存一个/多个子图，子图对象可以绘制具体的数据和设置子图属性
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        # axes同意转换为列表
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数 置子图的轴标签、轴范围、刻度和图例等属性
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点 实时更新图标
        # y是否可迭代 不可转换为列表
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
        # 确保x与y的个数一致
            x = [x] * n
        # 为空初始化
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        # enumerate为同时迭代数据和索引
        for i, (a, b) in enumerate(zip(x, y)):
            # （a,b) 都不空加入图标
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        # 清除子图 以便更新数据后重新绘制
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            #绘制子图
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        # 用于清除输出，以便在动画中实现数据的实时更新和显示
        display.clear_output(wait=True)