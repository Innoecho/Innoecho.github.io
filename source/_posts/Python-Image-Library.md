---
title: Python Image Library
date: 2017-04-25 22:35:58
tags: [Python, PIL]
categories: Python
---

Python图像库PIL的安装以及问题
<!--more-->

## 下载

在[这里](http://www.pythonware.com/products/pil/)下载PIL源代码

## 安装

### 下载依赖库

PIL需要依赖一些图像库，执行以下代码安装依赖库

```bash
sudo apt-get install libjpeg62-dev
sudo apt-get install zlib1g-dev
sudo apt-get install libfreetype6-dev
sudo apt-get install liblcms1-dev
```

### build

解压下载文件，并进入目录

执行`python setup.py build_ext -i`进行build并给出环境报告，如下所示

```bash
----------------------------------------------------------------
PIL 1.1.7 SETUP SUMMARY
----------------------------------------------------------------
*** TKINTER support not available (Tcl/Tk 8.5 libraries needed)
--- JPEG support available
--- ZLIB (PNG/ZIP) support available
--- FREETYPE support available
----------------------------------------------------------------
```

执行`python selftest.py`进行安装前的测试

如果上述操作没有任何错误，则可以执行`python setup.py install`进行最后的安装

## TroubleShooting

### decoder jpeg not available

如果出现以上错误，首先确认是否安装了相应的依赖库。

如果确定依赖库安装成功后仍有上述错误，则有可能是PIL无法定位相应的依赖库，此时执行以下代码即可

```bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
sudo ln -s /usr/lib/x86_64-linux-gnu/libfreetype.so /usr/lib
sudo ln -s /usr/lib/x86_64-linux-gnu/libz.so /usr/lib
```