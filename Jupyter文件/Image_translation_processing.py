import torch
import torchvision
import os
os.environ ["KMP_DUPLICATE_LIB_OK"] ="TRUE"
import matplotlib.pyplot as plt
from torch.utils import data
# torchvision 包含一些计算机视觉的相关库 transfrom包含图像处理的一些算法
from torchvision import transforms
from d2l import torch as d2l
# 图片以svg的格式进行显示
d2l.use_svg_display()
# 将PIL（Python Imaging Library）图像或NumPy数组转换为PyTorch张量（torch.Tensor）的格式
trans = transforms.ToTensor()
# 便用户下载、加载和准备Fashion MNIST数据集并将其转换为PyTorch的Dataset对象，以便与DataLoader一起使用进行批量数据加载
mnist_train = torchvision.datasets.FashionMNIST(
    root="T://ProgramsData//data", train=True, transform=trans, download=False)
mnist_test = torchvision.datasets.FashionMNIST(
    root="T://ProgramsData//data", train=False, transform=trans, download=False)
len(mnist_train), len(mnist_test)
first_sample_img,first_sample_lable = mnist_train[0]
# main_train二维数组 图片 标签
mnist_train[0][0].shape
def get_fashion_mnist_labels(labels):
    # 返回fashion_mnist数据集的文本标签
    text_labels =  ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
# 要显示的图像列表 行数 列数 缩放大小
def show_images(imgs,num_rows,num_cols,titles=None,scale = 1.5):
    # 根据行数 列数 和 缩放因子计算图像的大小
    figsize = (num_cols * scale,num_rows * scale)
    # subplots函数创建一个网格子图
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    # 二维子图数组展平成一维数组  子图 (axes) 被解包并存储在变量 axes 中
    axes = axes.flatten()
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imgshow(img)
        # 隐藏当前子图 (ax) 上的x轴刻度和标签
        ax.axes.get_xaxis().set_visible(False)
        # 隐藏当前子图 (ax) 上的y轴刻度和标签
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    # 在遍历所有图像和子图后，函数返回子图数组
    return axes
# torch加载数据集的一个batch，X为图像样本，y为对应标签
X,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
# 18张28*28格式的图片
show_images(X.reshape(18,28,28),2,9,titles=get_fashion_mnist_labels(y));

batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="T://ProgramsData//data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="T://ProgramsData//data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break


a = [16,32,64,128,256,512,1024,2048]
speed = []
def read_speed_test():
    for batch_size in a:
         train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers())
        
         timer = d2l.Timer()
         for X,y in train_iter:
            continue
         speed.append(round(float(timer.stop()),2))

read_speed_test()

print(speed)
import matplotlib.pyplot as plt
plt.plot(a,speed)
# 解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.ylabel('读取时间 /ms')
plt.xlabel('batch_size')
plt.show()