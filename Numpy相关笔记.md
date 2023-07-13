---
title: numpy学习笔记
date: 2023-07-08
---

## 0. ndarray对象

创建一个ndarray只需要调用Numpy中的array函数：

```python
numpy.array(object,dtype = None,copy=True,order=None,subok=false,ndim=0)
```

参数说明

| 名称   | 描述                                                      |
| ------ | --------------------------------------------------------- |
| object | 数组 或 嵌套的数列                                        |
| dtype  | 数组元素的数据类型（可选）                                |
| copy   | 对象是否需要复制（可选）                                  |
| order  | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
| subok  | 默认返回一个与基类类型一致的数组                          |
| ndmin  | 指定生成数组的最小维度（1开始）                           |



## 1. numpy的数据类型

支持的数据类型比python内置的类型要多，可以和c语言的数据类型对应上

数据类型对象（dtype）

```python
dt = np.dtype[('类型字段名'，数据类型)，('类型字段名2'，数据类型)...]
a = np.array([(19,),(20,),(21,)],dtype = dt)
```



## 2. numpy的数组属性

| 属性名     | 说明                                     |
| ---------- | ---------------------------------------- |
| ndim       | 秩，轴的数量或**数组的维度**             |
| shape      | 数组的维度，n行m列                       |
| size       | 数组元素的总个数，n*m个                  |
| dtype      | 包含的元素的数据类型                     |
| itemsize   | 每个元素多少字节                         |
| flags      | 对象的内存信息                           |
| real、imag | 元素的实部 虚部（对complex类型数据而言） |

**示例**

```python
import numpy as np
a = np.array([
    [1,2,3],
    [4,5,6]
])
print(a.shape) #输出（2，3）
b  = a.reshape(3,2) #单纯reshape并不会改变a的形状
print(b) #输出[[1,2],[3,4],[5,6]]
```

**注意：**

ndarray.reshape 通常返回的是非拷贝副本，即改变返回后数组的元素，原数组对应元素的值也会改变。（这里应该是类似于c当中指针的复制，b只是对a的一个引用）

```python
a  =  np.array([
    [1,2,3],
    [4,5,6]
])
print(a.shape)
b = a.reshape(3,2)
b[0][0] = 100
print(b)
print(a)
#这里a b同时改变了
output:
    [[100   2]
     [  3   4]
     [  5   6]]
    [[100   2   3]
     [  4   5   6]]
```



## 3. 创建数组

### 3.1 numpy.empty()

创建一个指定形状（shape，[维度]）、数据类型（shape）且未初始化的数组,**值是随机初始化的**

```python
numpy.empty(shape,dtype = float,order='C')
```

shape为形状如3*4维的可以将shape用[3,4]替代

dtype为数据类型

order为'C'行优先 ’F'列优先

### 3.2 numpy.ones()

以元素1来填充

```python
numpy.ones(shape,dtype = None,order = 'C')
```

### 3.3 numpy.zeros_like()

创建一个与给定数组具有相同形状的数组，以元素0来填充

```python
numpy.zeros_like(模型数组，dtype=None,order='K',subok=True,shape=None)
```

order表示存储数据，有行（‘C’）、有列（‘F’）以及模型数组‘K’

### 3.4 numpy.ones_like()

创建一个与给定数组具有相同形状的数组，以元素1来填充，参数解释同3.3

```python
numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
```



## 4. 从已有数组创建数组

### 4.1 numpy.asarray

```python
arr = numpy.asarray(a，dtype = None, order = None)
```

**这里的a可以是任意形式的参数，可以是列表、列表的元组、元组、元组的元组、元组的列表、多维数组,arr同a当中的数据一致**

### 4.2 numpy.frombuffer

```python
numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
```

接受buffer输入参数，以流的形式读入转化成ndarray对象

> 注意：*buffer 是字符串的时候，Python3 默认 str 是 Unicode 类型，所以要转成 bytestring 在原 str 前加上 b。*

示例：

```python
s = b'Hello,world'
a = np.frombuffer(s,dtype='S1')
print(a)
```

## 5. NumPy 从数值范围创建数组

### numpy.arange

```python
numpy.arange(start,stop,step,dtype)
```

| 参数  | 描述                                    |
| ----- | --------------------------------------- |
| start | 起始值，默认为0                         |
| stop  | 终止值，左闭右开，不包含                |
| step  | 步长，默认1                             |
| dtype | 返回ndarray的数据类型，默认输入数据类型 |

### numpy.linspace

等差数列，返回从start开始到stop结束，num个组成的等差数列，**此处默认包含stop**

```python
np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```

num为数列元素个数，endpoint是否包含最后一个元素

### numpy.logspace

创建一个等比数列，用法同上

```python
np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
```

参数说明：

| 参数  | 描述                           |
| ----- | ------------------------------ |
| start | 以base^start作为开始节点       |
| stop  | 以base^end作为结束节点         |
| num   | 要生成等长的样本数量，默认为50 |
| base  | 以多少作为底数                 |

```python
import numpy as np
a = np.logspace(0,9,10,base=2)
print (a)
#output
[  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```

## 6. 切片和索引

### slice函数

```python
a = np.arange(10)
s = slice(2,7,2)
print(a[s])
#output
#[2,4,6]
```

 ### 冒号进行切片

**start:stop:step**

```python
a[1:] #从1开始往后切
对于多维数组也同样适用
```



## 7. 高级索引

### 整数数组索引

**理解** 这里应该是取第0维的元素为坐标，然后到对应的ndarray当中取相关元素

```python
x = np.array([[1,  2],  [3,  4],  [5,  6]]) 
y = x[[0,1,2],  [0,1,0]]  
print (y)
#output
#[1,4,5]
```

```python
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print ('我们的数组是：' )
print (x)
print ('\n')
rows = np.array([[0,0],[3,3]]) 
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]  
print  ('这个数组的四个角元素是：')
print (y)
# output
#这个数组的四个角元素是：
#[[ 0  2]
# [ 9 11]]
```

### 借助切片`:` 和 `...`与索引数组组合

```python
a = np.array([[1,2,3], [4,5,6],[7,8,9]])
b = a[1:3, 1:3]
c = a[1:3,[1,2]]
d = a[...,1:]
print(b)
print(c)
print(d)
```

```python
[[5 6]
 [8 9]]
[[5 6]
 [8 9]]
[[2 3]
 [5 6]
 [8 9]]
```

**理解**

a[1:3]是指ndarray a取第0维第1、2两行坐标，然后再取第1维[1,2]位置的元素

### 布尔索引

布尔索引通过布尔运算（如：比较运算符）来获取符合**指定条件的元素的数组。**  `[]`当中可以是一个布尔运算

```python
#获取数组中大于5的所有元素
a = np.arange(0,10,2)
x = a[a>5]
print(x)
#output
#[6,8]
```

### 花式索引

利用整数数组进行索引

#### 一维数组

```python
x = np.arange(9)
print(x)
x2 = x[[0,6]]
print(x2)
#output
#[0,6]
```

#### 二维数组

axis = 0 和 axis = 1,单出现类似于x[[...]]的情况表示取axis = 0对应的那个轴

多个索引数组的情况

```python
x=np.arange(32).reshape((8,4))
print (x[np.ix_([1,5,7,2],[0,3,1,2])])
# 这里的np.ix_([1,5,7,2],[0,3,1,2])相当于求笛卡尔积结果为
# (1,0),(1,3),(1,1),(1,2)...然后取对应位置的元素
```





## 8. Numpy广播

广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式,对数组的算术运算通常在相应的元素上进行。

正常情况下，numpy数组之间的四则运算是建立在对应元素的情况下，当两个数组形状不一致时，会触发广播机制

```python
a = np.array([[ 0, 0, 0],
           [10,10,10],
           [20,20,20],
           [30,30,30]])
b = np.array([0,1,2])
print(a + b)
#output
#[[ 0  1  2]
# [10 11 12]
# [20 21 22]
# [30 31 32]]
```

![](https://cdn.staticaly.com/gh/SisyphusTang/Picture-bed@master/20230701/numpy广播机制.4brjhzrp6qg0.png)

numpy广播的规则：

+ 输入数组之间要有一个维度长度相同（或者有个维度是1），否则无法广播
+ 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐

### tile()函数

```python
numpy.tile(A,reps)
```

这里的 A 就是数组，reps 可以是一个数，一个列表、元组或者数组等，就是类数组的类型。先要理解准确，先把 A 当作一个块（看作一个整体，别分开研究每个元素）。

+ 如果 reps 是一个数，就是简单的将 A 向右复制 reps - 1 次形成新的数组，就是 reps 个 A 横向排列
+ 如果 reps 是一个 array-like（类数组的，如列表，元组，数组）类型的，它有两个元素，如 [m , n]，实际上就是将 A 这个块变成 m * n 个 A 组成的新数组，有 m 行，n 列 A



## 9. 迭代数组

### 迭代对象nditer

```python
a = np.arange(6).reshape(2,3)
for x in np.nditer(a):
    print(x)
#output:0,1,2,3,4,5
```

控制遍历顺序

+ for x  in np.nditer(a,order='F') 不改变存储顺序  
+ for x in np.nditer(a,order='C')  不改变存储顺序

a 和 a.T (a的转置)的遍历顺序是一样的

### 修改数组当中元素的值

可选参数`op_flags`设置为`readwrite`或`writeonly`

```python
a = np.arange(20)
for x in np.nditer(a,op_flags=['readwrite']):
    x[...] = 2*x
#之后a当中内容修改为原来的两倍
```

numpy.copy 做了特殊处理，它拷贝的时候不是直接把对方的内存复制，而是按照上面 order 指定的顺序逐一拷贝。



## 10. numpy数组操作

### 修改数组形状

#### reshape

```python
numpy.reshape(arr,newshape,order='C')
```

arr为修改形状的数组，newshape为整数或整数数组，需要兼容原有的形状

order：‘C’按行，‘F’按列

#### numpy.ndarray.flat

数组元素迭代器

```python
a = numpy.arange(xxx)
for element in a.flat:
    print(element)#按照行的顺序
```

#### numpy.ndarray.flatten

返回一份数组拷贝，对拷贝做的修改不会影响原数组

```python
ndarray.flatten(order='C')
```

order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'K' -- 元素在内存中的出现顺序

#### numpy.ravel

展平的数组元素，顺序通常是"C风格"，返回的是数组视图（view，有点类似 C/C++引用reference的意味），**修改会影响原始数组。**

```python
numpy.ravel(a, order='C')
```

### 翻转数组

### 修改数组维度

###  连接数组

### 分割数组

### 数组元素的添加与删除

| 函数   | 作用                                     |
| ------ | ---------------------------------------- |
| resize | 返回指定形状的新数组                     |
| append | 将值添加到数组末尾                       |
| insert | 沿指定轴将值插入到指定下标之前           |
| delete | 删掉某个轴的子数组，并返回删除后的新数组 |
| unique | 查找数组内唯一元素                       |

### numpy.resize

函数返回指定大小的新数组，如果新数组的大小比原数组大，则会有一部分的重合部分。

```python
numpy.resize(arr,shape)
```

+ arr修改的数组 如3*4维的  shape就用（3，4）代替
+ shape返回数组的情况

#### numpy.append

在数组的某一个维度上添加一个新数组。输入数组的维度必须匹配。

```python
a = np.array([[1,2,3],[4,5,6]])
 
print ('第一个数组：')
print (a)
print ('\n')
#output
#第一个数组：
#[[1 2 3]
# [4 5 6]]

print ('向数组添加元素：')
print (np.append(a, [7,8,9]))
print ('\n')
#output
#向数组添加元素：
#[1 2 3 4 5 6 7 8 9]
 
print ('沿轴 0 添加元素：')
print (np.append(a, [[7,8,9]],axis = 0))
print ('\n')
#沿轴 0 添加元素：
#[[1 2 3]
# [4 5 6]
# [7 8 9]]
 
print ('沿轴 1 添加元素：')
print (np.append(a, [[5,5,5],[7,8,9]],axis = 1))
#沿轴 1 添加元素：
#[[1 2 3 5 5 5]
# [4 5 6 7 8 9]]
```

#### numpy.insert

```python
numpy.insert(arr, obj, values, axis)
```

+ arr为插入对象
+ obj为插入位置（当前位置元素往后面挪）
+ values要插入的值
+ axis沿着它插入的轴找位置，没有这个参数数组将会被展开

```python
a = np.array([[1,2],[3,4],[5,6]])
 
print ('第一个数组：')
print (a)
print ('\n')
 
print ('未传递 Axis 参数。 在删除之前输入数组会被展开。')
print (np.insert(a,3,[11,12]))
print ('\n')
print ('传递了 Axis 参数。 会广播值数组来配输入数组。')
 
print ('沿轴 0 广播：')
print (np.insert(a,1,[11],axis = 0))
print ('\n')
 
print ('沿轴 1 广播：')
print (np.insert(a,1,11,axis = 1))
```

#### numpy.delete

返回从输入数组中删除指定子数组的新数组

```python
numpy.delete(arr,obj,axis)
```

+ arr:输入数组
+ obj:可以被切片，整数或者整数数组（坐标），从原数组当中切除的片
+ axis:沿着哪个轴找对应数组，如果没有默认展开进行删除

```python
a = np.arange(12).reshape(3,4)
 
print ('第一个数组：')
print (a)
print ('\n')
 
print ('未传递 Axis 参数。 在插入之前输入数组会被展开。')
print (np.delete(a,5))
print ('\n')
# [ 0  1  2  3  4  6  7  8  9 10 11]
print ('删除第二列：')
print (np.delete(a,1,axis = 1))
print ('\n')
# [[ 0  2  3]
# [ 4  6  7]
# [ 8 10 11]]

print ('包含从数组中删除的替代值的切片：')
a = np.array([1,2,3,4,5,6,7,8,9,10])
print (np.delete(a, np.s_[::2]))
#[ 2  4  6  8 10]
```



#### numpy.unique

用于去除数组中的重复元素

```python
numpy.unique(arr,return_index,return_inverse,return_counts)
```

- `arr`：输入数组，如果不是一维数组则会展开
- `return_index`：如果为`true`，返回新列表元素在旧列表中的位置（下标），并以列表形式储
- `return_inverse`：如果为`true`，返回旧列表元素在新列表中的位置（下标），并以列表形式储
- `return_counts`：如果为`true`，返回去重数组中的元素在原数组中的出现次数

```python
a = np.array([5,2,6,2,7,5,6,8,2,9])
u = np.unique(a) # u为[2 5 6 7 8 9]
#indices存储新列表元素在旧列表当中的位置[1 0 2 4 7 9]
u,indices = np.unique(a, return_index = True) 
#indices存储旧列表元素在新列表当中的位置
u,indices = np.unique(a,return_inverse = True)
#indices存储去重元素的重复数量
u,indices = np.unique(a,return_counts = True)
```

## 11. numoy位运算

[具体参见此处](https://www.runoob.com/numpy/numpy-binary-operators.html)

| 函数          | 描述                   |
| :------------ | :--------------------- |
| `bitwise_and` | 对数组元素执行位与操作 |
| `bitwise_or`  | 对数组元素执行位或操作 |
| `invert`      | 按位取反               |
| `left_shift`  | 向左移动二进制表示的位 |
| `right_shift` | 向右移动二进制表示的位 |



## 12. numpy字符串函数

以下函数用于对 dtype 为 numpy.string_ 或 numpy.unicode_ 的数组执行向量化字符串操作。 它们基于 Python 内置库中的标准字符串函数。这些函数在字符数组类（numpy.char）中定义。

| 函数           | 描述                                       |
| :------------- | :----------------------------------------- |
| `add()`        | 对两个数组的逐个字符串元素进行连接         |
| multiply()     | 返回按元素多重连接后的字符串               |
| `center()`     | 居中字符串                                 |
| `capitalize()` | 将字符串第一个字母转换为大写               |
| `title()`      | 将字符串的每个单词的第一个字母转换为大写   |
| `lower()`      | 数组元素转换为小写                         |
| `upper()`      | 数组元素转换为大写                         |
| `split()`      | 指定分隔符对字符串进行分割，并返回数组列表 |
| `splitlines()` | 返回元素中的行列表，以换行符分割           |
| `strip()`      | 移除元素开头或者结尾处的特定字符           |
| `join()`       | 通过指定分隔符来连接数组中的元素           |
| `replace()`    | 使用新字符串替换字符串中的所有子字符串     |
| `decode()`     | 数组元素依次调用`str.decode`               |
| `encode()`     | 数组元素依次调用`str.encode`               |

## 13. numpy数学函数

### 三角函数

sin()、cos()、tan() 函数体内都是弧度，角度在使用的时候要进行一个转换

```python
a = np.array([0,30,60,90])
#转换为弧度
t = a*np.pi/180
print(np.sin(t))
print(np.cos(t))
print(np.tan(t))
```

  同时，弧度可以通过`np.degree`转换为角度

### 舍入函数

```python
numpy.around(a,decimals)
```

+ a 为数组
+ decimals：舍入的小数的位数，默认值为0，为负代表从小数点左边开始

```python
a = np.array([1.0,5.55,  123,  0.567,  25.532])  
#[ 1.    5.6 123.    0.6  25.5]
print (np.around(a, decimals =  1))
#[  0.  10. 120.   0.  30.]
print (np.around(a, decimals =  -1))
```

### numpy.floor(）

返回小于或者等于指定表达式的最大整数，即向下取整

```python
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])
#[-2.  1. -1.  0. 10.]
print (np.floor(a))
```

 ### numpy.ceil()

返回大于或者等于指定表达式的最小整数，即向上取整

```python
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])  
print (np.ceil(a))
# [-1.  2. -0.  1. 10.]
```



## 14. numpy算术函数

### 加、减、乘、除

add()、subtract()、multiply、divide()

注意：这里的加减乘除是针对形状类似的两个数组之间的运算，即对应位置上的数乘对应位置上的数，当然，如果形状不一至，符合条件的话会进行广播

### numpy.reciprocal()

返回参数逐元素的倒数

### numpy.power()

`numpy.power(a,b)` a为底数，b为幂的新数组

```python
a = np.array([10,100,1000])
#[100 10000 1000000]
print(np.power(a,2))
```

### numpy.mod()

`numpy.mod(a,b)`计算数组`a`与数组`b`mod之后的余数



## 15. numpy统计函数

### numpy.amind() 与 numpy.amax()

`amind()`用于计算数组中元素沿指定轴的最小值

`amax()`用于计算数组中的元素沿指定轴的最大值

**一个3\*4的矩阵图如下所示**

![](https://cdn.staticaly.com/gh/SisyphusTang/Picture-bed@master/20230703/axis=0.44j07nobmig0.webp)

```python
a = np.array([[3,7,5],[8,4,3],[2,4,9]])  
print (np.amin(a,1)) #[3 3 2]横着最小的
print (np.amin(a,0)) #[2,4,3]竖着最小的
print (np.amax(a, axis =  0))#[8,7,9]竖着最大的
```

### numpy.ptp()

计算数组中元素最大值与最小值的差

### numpy.percentile()

**找到一个数（随机？），在某一维度（默认整个数组）上，有`%q`的数比它小，有`%(1-q)`的数据比它大**

```python
numpy.percentile(a, q, axis)
```

+ a:输入数组
+ q:要计算的百分位数 0-100之间
+ axis：沿着它计算的百分位数的轴

示例：

```python
a = np.array([[10, 7, 4], [3, 2, 1]])
print ('我们的数组是：')
print (a)
#[[10  7  4]
# [ 3  2  1]]
print ('调用 percentile() 函数：')
# 50% 的分位数，就是 a 里排序之后的中位数
print (np.percentile(a, 50)) 
#3.5 有一半的数比3.5小 有一半的数据比3.5大
# axis 为 0，在纵列上求
print (np.percentile(a, 50, axis=0)) 
#[6.5 4.5 2.5] axis=0的那个列上有50%比它大 有50%比它小的情况 
# axis 为 1，在横行上求
print (np.percentile(a, 50, axis=1)) 
#axis = 1 [7. 2.]的那个维度情况 
# 保持维度不变
print (np.percentile(a, 50, axis=1, keepdims=True))
#[[7.]
# [2.]]
```

### numpy.median()

用于计算数组a中元素的中位数

```python
np.median(a,axis = ?)
```

### numpy.mean()

计算数组元素的算术平均值，算术平均值是沿轴的元素的总和除以元素的数量。

```python
np.mean(a,axis = ?)
```

### numpy.average()

计算数组的加权平均数

考虑数组[1,2,3,4]和相应的权重[4,3,2,1]，通过将相应元素的乘积相加，并将和除以权重的和，来计算加权平均值。

```python
#arr是原数组 weights是权重数组
np.average(arr,weights=wts)
```

### 标准差

```python
np.std(arr)
std = sqrt(mean((x - x.mean())**2))
```

### 方差

```
np.var()
```





## 16.  numpy排序 条件筛选函数

### numpy.sort()

```python
numpy.sort(a, axis, kind, order)
```

参数说明：

- a: 要排序的数组
- axis: 沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序， axis=0 按列排序，axis=1 按行排序
- kind: 默认为'quicksort'（快速排序）
- order: 如果数组包含字段，则是要排序的字段

```python
# 在 sort 函数中排序字段 
dt = np.dtype([('name',  'S10'),('age',  int)]) 
a = np.array([("raju",21),("anil",25),("ravi",  17),  ("amar",27)], dtype = dt)  
#按字母升序来排序
print (np.sort(a, order =  'name'))
```

| 种类                      | 速度 | 最坏情况      | 工作空间 | 稳定性 |
| :------------------------ | :--- | :------------ | :------- | :----- |
| `'quicksort'`（快速排序） | 1    | `O(n^2)`      | 0        | 否     |
| `'mergesort'`（归并排序） | 2    | `O(n*log(n))` | ~n/2     | 是     |
| `'heapsort'`（堆排序）    | 3    | `O(n*log(n))` | 0        | 否     |

### numpy.argsort()

返回数组从小到大排序后的索引值

```python
np.argsort(arr)
```

### numpy.lexsort()

numpy.lexsort() 用于对多个序列进行排序。先排第一列，第一列优先级一样再排第二列

```python
nm =  ('raju','anil','ravi','amar') 
dv =  ('f.y.',  's.y.',  's.y.',  'f.y.') 
print ([nm[i]  +  ", "  + dv[i]  for i in ind])
#先排nm 再排dv
#['amar, f.y.', 'anil, s.y.', 'raju, f.y.', 'ravi, s.y.']
```

### numpy.argmax()与numpy.argmin()

numpy.argmax() 和 numpy.argmin()函数分别沿给定轴返回最大和最小元素的索引

### numpy.nonzero()

返回输入数组当中非零元素的索引

### numpy.where()

返回输入数组中满足给定条件的元素的索引

**where()当中是一个布尔表达式**

```python
x = np.arange(9.).reshape(3,  3)  
y = np.where(x >  3)  
print (y)
```

### numpy.extract()

numpy.extract() 函数根据某个条件从数组中抽取元素，返回满条件的元素。同理，extract()当中存放了一个数组，这个数组存放对每个元素的true或者false

```python
x = np.arange(9.).reshape(3,  3)  
print ('我们的数组是：')
print (x)
# 定义条件, 选择偶数元素
condition = np.mod(x,2)  ==  0  
print ('按元素的条件值：')
print (condition)
print ('使用条件提取元素：')
print (np.extract(condition, x))
```

```python
#输出结果
我们的数组是：
[[0. 1. 2.]
 [3. 4. 5.]
 [6. 7. 8.]]
按元素的条件值：
[[ True False  True]
 [False  True False]
 [ True False  True]]
使用条件提取元素：
[0. 2. 4. 6. 8.]
```

## 17. Numpy字节交换

参见：

https://www.runoob.com/numpy/numpy-byte-swapping.html



## 18.numpy的副本和视图

+ 副本：对原始数据的拷贝，副本的修改**不会影响到原始数据**，使用ndarray的copy()返回一个副本
+ 视图：对数据的一个引用，ndarray的view()返回的数据修改不会影响到原始数据，ndarray的切片返回的数据修改会影响到原始数据

看一下id是否相同，id相同说明指向的是同一个位置，id不同，说明两个数据是相互独立的，不会相互影响到。



## 19. numpy矩阵库（Matrix）

### 矩阵的转置

```python
a = np.arange(6).reshape(2,3)
b = a.T
```

### matlib.empty()

返回一个新矩阵，数据是随机填充的。

```python
numpy.matlib.empty(shape,dtype,order)
```

- **shape**: 定义新矩阵形状的整数或整数元组
- **Dtype**: 可选，数据类型
- **order**: C（行序优先） 或者 F（列序优先）

### numpy.matlib.zeros()

创建一个以0填充的矩阵

```python
np.matlib.zeros((2,2))
```

### numpy.matlib.ones()

返回一个以1填充的矩阵

```python
np.matlib.ones((2,2))
```

### numpy.matlib.eye()

返回一个对角线为1，其它位置为0的矩阵

```python
numpy.matlib.eye(n,M,k,dtype)
```

- **n**: 返回矩阵的行数
- **M**: 返回矩阵的列数，默认为 n
- **k**: 对角线的索引
- **dtype**: 数据类型

### numpy.matlib.identity()

numpy.matlib.identity() 函数返回给定大小的单位矩阵。

```python
print (np.matlib.identity(5, dtype =  float))
```

### numpy.matlib.rand()

numpy.matlib.rand() 函数创建一个给定大小的矩阵，数据是随机填充的。

## 20. numpy线性代数

| 函数          | 描述                                               |
| :------------ | :------------------------------------------------- |
| `dot`         | 两个数组的点积，二维是计算向量积，即元素对应相乘。 |
| `vdot`        | 两个向量的点积                                     |
| `inner`       | 两个数组的内积                                     |
| `matmul`      | 两个数组的矩阵积                                   |
| `determinant` | 数组的行列式                                       |
| `solve`       | 求解线性矩阵方程                                   |
| `inv`         | 计算矩阵的乘法逆矩阵                               |

## 21. numpy IO

- load() 和 save() 函数是读写文件数组数据的两个主要函数，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npy 的文件中。
- savez() 函数用于将多个数组写入文件，默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为 .npz 的文件中。
- loadtxt() 和 savetxt() 函数处理正常的文本文件(.txt 等)

### numpy.save()

numpy.save() 函数将数组保存到以 .npy 为扩展名的文件中。

```
numpy.save(file, arr, allow_pickle=True fix_imports=True)
```

- **file**：要保存的文件，扩展名为 .npy，如果文件路径末尾没有扩展名 .npy，该扩展名会被自动加上。
- **arr**: 要保存的数组
- **allow_pickle**: 可选，布尔值，允许使用 Python pickles 保存对象数组，Python 中的 pickle 用于在保存到磁盘文件或从磁盘文件读取之前，对对象进行序列化和反序列化。
- **fix_imports**: 可选，为了方便 Pyhton2 中读取 Python3 保存的数据。

### np.savez

numpy.savez() 函数将多个数组保存到以 npz 为扩展名的文件中。

```
numpy.savez(file, *args, **kwds)
```

参数说明：

- **file**：要保存的文件，扩展名为 **.npz**，如果文件路径末尾没有扩展名 **.npz**，该扩展名会被自动加上。
- **args**: 要保存的数组，可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为 **arr_0**, **arr_1**, …　。
- **kwds**: 要保存的数组使用关键字名称。

### savetxt()

savetxt() 函数是以简单的文本文件格式存储数据，对应的使用 loadtxt() 函数来获取数据。

```
np.loadtxt(FILENAME, dtype=int, delimiter=' ')
np.savetxt(FILENAME, a, fmt="%d", delimiter=",")
```



## 22. NumPy Matplotlib

matplotlib是python的绘图库

`Anaconda navigator` -> `enviroment`->`not install` 进行下载安装

matplotlib当中最常用的pyplot子模块

```python
import matplotlib.pyplot as plt
```

### pyplot基本用法

#### 绘制简单的线图

基本代码为`plt.plot(x,y,format_string,**kwargs)`

+ x、y为所对应轴的数据，是列表或者数组
+ format_string是控制曲线的格式字符串
+ 更多的（x,y,format_string）数据

```python
x = [0,1,2,3,4,5]
y = [0,1,4,9,16,25]
plt.plot(x,y)
plt.show()
```

### 绘制散点图

**scatter()函数：**
散点图的绘制工具，基本格式为：

```python
plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None,vmin=None, vmax=None, alpha=None, linewidths=None,verts=None, edgecolors=None, hold=None, data=None,**kwargs)
```

其中：

x，y：输入数据，形状为 shape(n，)）的数组。

c：标记的颜色，可选，默认为'b'即蓝色。

marker：标记的样式，默认为'o'。

alpha：透明度，实数,0～1。

linewidths：标记点的宽度。
### 绘制条形图

#### bar()函数：

绘制条形图的工具，基本格式为：

```python
bar(x，height， width=0.8，bottom=None，hold=None，data=None，\**kwargs)
```

x：x轴刻度，为数值序列或字符串序列

height：y轴，展示的数据→柱形图高度

### 绘制饼图

#### pie()函数：

饼图绘制工具，基本格式为：

```
plt.pie(values, labels=labels, colors=colors, explode=explode, autopct=autopct, shadow=shadow)
```

values：表示饼图的数值，可以是一个序列或者列表；
labels：表示饼图各部分的标签，可以是一个序列或者列表；
colors：表示饼图各部分的颜色，可以是一个序列或者列表；
explode：表示饼图各部分与中心的距离，可以是一个序列或者列表；
autopct：表示饼图各部分所占比例的显示方式；
shadow：表示是否给饼图添加阴影。
