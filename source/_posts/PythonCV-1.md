---
title: Programming Computer Vision with Python Chapter 1
date: 2017-05-07 15:42:48
tags: [CV, Python, PIL]
categories: CV
---

基本的图像操作和处理
<!-- more -->


# Chapter 1. 基本的图像操作和处理

## 1.1 PIL

 库的导入以及图像的显示方法


```python
from PIL import Image
```


```python
pil_im = Image.open('../data/empire.jpg')
pil_im.show()
```

### 1.1.1 转换图像格式

使用`save()`方法保存时，PIL会根据文件扩展名判断图像的格式


```python
import imtools

filelist = imtools.get_imlist("../data")
for file in filelist:
    print(file)
```

    ../data/fisherman.jpg
    ../data/Univ3.jpg
    ../data/sf_view2.jpg
    ../data/boy_on_hill.jpg
    ../data/Univ1.jpg
    ../data/alcatraz1.jpg
    ../data/sf_view1.jpg
    ../data/alcatraz2.jpg
    ../data/sunset_tree.jpg
    ../data/climbing_2_small.jpg
    ../data/crans_1_small.jpg
    ../data/turningtorso1.jpg
    ../data/empire.jpg
    ../data/climbing_1_small.jpg
    ../data/Univ2.jpg
    ../data/Univ5.jpg
    ../data/Univ4.jpg
    ../data/crans_2_small.jpg


### 1.1.2 创建缩略图

使用`thumbnail`方法创建缩略图，例如


```python
pil_im.thumbnail((128, 128))
pil_im.show()
```

### 1.1.3 复制和粘贴图像区域

使用`crop()`方法从图像中裁剪指定区域


```python
pil_im = Image.open('../data/empire.jpg')

# 创建裁剪区域
box = (100, 100, 400, 400)
region = pil_im.crop(box)

# 旋转region以便于区分
region = region.transpose(Image.ROTATE_180)
pil_im.paste(region, box)

pil_im.show()
```

### 1.1.4 调整尺寸和旋转


```python
out = pil_im.resize((128, 128))
out.show()

out = pil_im.rotate(45)
out.show()
```

## 1.2 Matplotlib

### 1.2.1 绘制图像、点和线


```python
from PIL import Image
from pylab import *

# 读取图像到数组中
im = array(Image.open('../data/empire.jpg'))

figure(figsize = (10, 5))
# 绘制图像
imshow(im)

# 点
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

# 使用红色星状标记绘制点
plot(x, y, 'r*')

# 绘制连接线
plot(x[:2], y[:2])

# 添加标题，显示绘制的图像
title('Plotting: "empire.jpg"')
show()
```


![png](output_16_0.png)


### 1.2.2 图像轮廓和直方图


```python
from PIL import Image
from pylab import *

# 读取图像到数组中
im = array(Image.open('../data/empire.jpg').convert('L'))

# 新建图像
figure(figsize = (15,5))
gray()

# 在原点的左上角显示轮廓图像
subplot(1,2,1)
contour(im, origin = 'image')

axis('equal')
axis('off')

# 直方图
subplot(1,2,2)
hist(im.flatten(), 128)

# 显示图像
show()
```


![png](output_18_0.png)


### 1.2.3 交互式标注

代码如下所示：
```python
from PIL import Image
from pylab import *

im = array(Image.open('../data/empire.jpg'))
imshow(im)

print 'Please click 3 points'

x = ginput(3)

print 'you clicked :', x

show()
```
示例参见1_2_3.py

## 1.3 NumPy

Python科学计算工具包

### 1.3.1 图像数组表示
NumPy数组访问方式与Python类似


```python
from PIL import Image
from numpy import *

im = array(Image.open('../data/empire.jpg'))
print(im.shape, im.dtype)

im = array(Image.open('../data/empire.jpg').convert('L'), 'f')
print(im.shape, im.dtype)
```

    (800, 569, 3) uint8
    (800, 569) float32


### 1.3.2 灰度变换


```python
from PIL import Image
from numpy import *

figure(figsize = (17, 10))
gray()

im = array(Image.open('../data/empire.jpg').convert('L'))
subplot(2,4,1)
imshow(im)
subplot(2,4,5)
hist(im.flatten(), 128)

# 反相操作
im2 = 255 - im
subplot(2,4,2)
imshow(im2)
subplot(2,4,6)
hist(im2.flatten(), 128)

# 将图像像素值变换到100-200区间
im3 = (100.0/255) * im + 100
subplot(2,4,3)
imshow(im3)
subplot(2,4,7)
hist(im3.flatten(), 128)

# 对图像像素值求平方后得到的图像
im4 = 255.0 * (im/255.0)**2
subplot(2,4,4)
imshow(im4)
subplot(2,4,8)
hist(im4.flatten(), 128)

show()
```


![png](output_25_0.png)


### 1.3.3 图像缩放


```python
""" 图像缩放 """
def imresize(im, sz):
	pil_im = Image.fromarray(uint8(im))

	return array(pil_im.resize(sz))
```

### 1.3.4 直方图均衡化


```python
import imtools
from PIL import Image
from numpy import *

im = array(Image.open('../data/AquaTermi_lowcontrast.jpg').convert('L'))
im2, cdf = imtools.histeq(im)

figure(figsize = (15, 10))
gray()

subplot(2,2,1)
imshow(im)
subplot(2,2,3)
hist(im.flatten(), 128)

subplot(2,2,2)
imshow(im2)
subplot(2,2,4)
hist(im2.flatten(), 128)

show()
```


![png](output_29_0.png)


### 1.3.5 图像平均


```python
""" 图像平均 """
def compute_average(imlist):

	# 打开第一幅图像，将其存储在浮点型数组中
	averageim = array(Image.open(imlist[0]), 'f')

	for imname in imlist[1:]:
		try:
			averageim += array(Image.open(imname))
		except:
			print(imname + '...skipped')

	averageim /= len(imlist)

	# 返回uint8类型的平均图像
	return array(averageim, 'uint8')
```

### 1.3.6 图像的主成分分析（PCA）


```python
""" 主成分分析：
	输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
	返回：投影矩阵（按照维度的重要性排序）、方差和均值 """
def pca(X):
	# 获取维数
	num_data, dim = X.shape

	# 数据中心化
	mean_X = X.mean(axis = 0)
	X = X - mean_X

	if dim > num_data:
		# 使用紧致技巧

		# 协方差矩阵
		M = dot(X, X.T)
		# 特征值和特征向量
		e, EV = linalg.eigh(M)
		# 紧致技巧
		tmp = dot(X.T, EV).T

		# 由于最后的特征向量是我们所需要的，所以需要将其逆转
		V = tmp[::-1]
		# 由于特征值是按照递增顺序排列的，所以需要将其逆转
		S = sqrt(e)[::-1]

		for i in range(V.shape[1]):
			V[:, i] /= S
	else:
		# 使用SVD方法
		
		U, S, V = linalg.svd(X)
		# 仅仅返回前num_data维数据
		V = V[:num_data]

	# 返回投影矩阵、方差和均值
	return V, S, mean_X
```


```python
from PIL import Image
from numpy import *
from pylab import *
import pca
import imtools

# 获取图像列表
imlist = imtools.get_imlist("../data/a_thumbs")
# 打开一幅图像，获取大小
im = array(Image.open(imlist[0]))
m, n = im.shape[0:2]
# 获取图像数目
imnbr = len(imlist)

# 创建矩阵，保存所有图像数据
immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')

# 主成分分析
V, S, immean = pca.pca(immatrix)

# 显示结果
figure(figsize = (20, 10))
gray()
subplot(2,4,1)
# 显示均值
imshow(immean.reshape(m, n))
axis('off')
# 显示前7个模式
for i in range(7):
    subplot(2, 4, i+2)
    imshow(V[i].reshape(m,n))
    axis('off')
    
show()
```


![png](output_34_0.png)


### 1.3.7 保存数据

#### 使用pickle模块
使用pickle可以封装几乎所有的python对象

**封装**
```python
with open('font_pca_modes.pkl', 'wb') as f:
    pickle.dump(immean, f)
    pickle.dump(V, f)
```
**拆封**
```python
with open('font_pca_modes.pkl', 'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)
```

#### 使用NumPy的读写函数
如果数据结构不复杂，可以直接将其存为文本文件

```python
savetxt('test.txt', x, '%i')
x = loadtxt('test.txt')
```

## 1.4 Scipy

Scipy提供关于图像处理的许多功能模块

## 1.4.1 图像模糊


```python
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters

figure(figsize = (15,10))
gray()

im = array(Image.open('../data/empire.jpg').convert('L'))
sigma = [0, 2, 5, 10]

for i in range(4):
    subplot(1,4,i+1)
    imshow(filters.gaussian_filter(im, sigma[i]))
    axis('off')

show()
```


![png](output_38_0.png)


### 1.4.2 图像导数

描述图像强度变化的强弱


```python
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters

figure(figsize = (15,10))
gray()

im = array(Image.open('../data/empire.jpg').convert('L'))
subplot(1,4,1)
imshow(im)
axis('off')

# Sobel导数滤波器
imx = zeros(im.shape)
filters.sobel(im, 1, imx)
subplot(1,4,2)
imshow(imx)
axis('off')

imy = zeros(im.shape)
filters.sobel(im, 0, imy)
subplot(1,4,3)
imshow(imy)
axis('off')

magnitude = sqrt(imx**2 + imy**2)
subplot(1,4,4)
imshow(255-magnitude)
axis('off')

show()
```


![png](output_40_0.png)


为了在图像噪声方面更稳健以及在任意尺度上计算导数，我们可以使用高斯导数滤波器


```python
from PIL import Image
from pylab import *
from numpy import *
from scipy.ndimage import filters

figure(figsize = (15,10))
gray()

im = array(Image.open('../data/empire.jpg').convert('L'))
subplot(1,4,1)
imshow(im)
axis('off')

sigma = 10

# 高斯导数滤波器
imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
subplot(1,4,2)
imshow(imx)
axis('off')

imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
subplot(1,4,3)
imshow(imy)
axis('off')

magnitude = sqrt(imx**2)
subplot(1,4,4)
imshow(255-magnitude)
axis('off')

show()
```


![png](output_42_0.png)


### 1.4.3 形态学：对象计数

使用morphology和measurements模块


```python
from PIL import Image
# from pylab import *
# from numpy import *
from scipy.ndimage import measurements, morphology

# 读入图片
im = array(Image.open('../data/houses.png').convert('L'))
# 二值化操作
im = 1*(im < 128)

figure(figsize = (15, 10))
subplot(1,2,1)
imshow(im)
axis("off")

# 形态学计数
labels, nbr_obeject = measurements.label(im)

subplot(1,2,2)
imshow(labels)
axis("off")

# 显示结果
show()
print("Before opening operation, Number of objects:", nbr_obeject)

# 进行形态学开操作
im_open = morphology.binary_opening(im, ones((9, 5)), iterations=2)

figure(figsize = (15, 10))
subplot(1,2,1)
imshow(im_open)
axis("off")

# 形态学计数
labels_open, nbr_obeject_open = measurements.label(im_open)

subplot(1,2,2)
imshow(labels_open)
axis("off")

# 显示结果
show()
print("After opening operation, Number of objects:", nbr_obeject_open)

```


![png](output_44_0.png)


    Before opening operation, Number of objects: 45



![png](output_44_2.png)


    After opening operation, Number of objects: 48


### 1.4.4 一些有用的SciPy模块

#### io模块读写.mat文件

```python
data = scipy.io.loadmat('test.mat')

data = {}
data['x'] = x
scipy.io.savemat('test.mat', data)
```

#### misc模块将数组对象保存为图像形式

```python
scipy.misc.imsave('test.jpg', im)
```

## 1.5 图像去噪

图像去噪是在去除图像噪声的同时，尽可能地保留图像细节和结构的处理技术。

这里使用ROF(Rudin-Osher-Fatemi)去噪模型，代码如下所示：

```python
from numpy import *

""" 实现ROF去噪模型
	输入：含有噪声的灰度图像、U的初始值、停止条件、步长、TV正则项权值
	输出：去噪和去除纹理后的图像、纹理残留"""
def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
	# 噪声图像的大小
	m, n = im.shape

	# 初始化
	U = U_init
	# 对偶域的x分量
	Px = im
	# 对偶域的y分量
	Py = im
	error = 1

	while(error > tolerance):
		Uold = U

		# 原始变量的梯度
		# 变量U梯度的x分量
		GradUx = roll(U, -1, axis=1) - U
		# 变量U梯度的y分量
		GradUy = roll(U, -1, axis=0) - U

		# 更新对偶变量
		PxNew = Px + (tau/tv_weight)*GradUx
		PyNew = Py + (tau/tv_weight)*GradUy
		NormNew = maximum(1, sqrt(PxNew**2 + PyNew**2))

		# 更新x分量（对偶）
		Px = PxNew/NormNew
		# 更新y分量（对偶）
		Py = PyNew/NormNew

		# 更新原始变量
		# 对x分量进行向右x轴平移
		RxPx = roll(Px, 1, axis=1)
		# 对y分量进行向右y轴平移
		RyPy = roll(Py, 1, axis=0)

		# 对偶域的散度
		DivP = (Px-RxPx) + (Py-RyPy)
		# 更新原始变量
		U = im + tv_weight * DivP

		# 更新误差
		error = linalg.norm(U-Uold)/sqrt(m*n)

	return U, im-U
```

下面是合成的噪声图像示例


```python
from numpy import *
from numpy import random
from scipy.ndimage import filters
import rof

# 使用噪声创建合成图像
im = zeros((500, 500))
im[100:400, 100:400] = 128
im[200:300, 200:300] = 255
im_noise = im + 30*random.standard_normal((500, 500))

U, T = rof.denoise(im_noise, im_noise)

figure(figsize = (15, 10))
gray()

subplot(1,3,1)
imshow(im)
axis('off')

subplot(1,3,2)
imshow(im_noise)
axis('off')

subplot(1,3,3)
imshow(U)
axis('off')

show()
```


![png](output_49_0.png)
