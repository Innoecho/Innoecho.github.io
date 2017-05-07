---
title: 使用Hexo建立Blog
date: 2017-03-02 16:15:50
tags: [Hexo, Blog]
categories: Hexo
---

Hexo是一款基于Node.js的静态博客框架，用Hexo搭建博客可以自由配置需要的功能、界面，所以闲着没事也自己搭了个。以下所有操作针对Ubuntu系统，当然其他操作系统大体类似。
<!--more-->

# 安装必要工具

## Node.js

首先安装Node.js，在[官网](https://nodejs.org/en/)下载相对应的版本，安装即可

## Hexo

命令行下执行`npm install -g hexo-cli`即可

Hexo用来根据配置文件生成静态页面

## Git

命令行下执行`sudo apt-get install git`即可

如果没有GitHub账号的话，还需要申请Github账号，这里Github是用来托管生成的静态页面的

#  配置

## 建立Github仓库

在Github上建立名为"name.github.io"的仓库，固定写法，name自取，这个仓库所存储的静态页面可以通过域名https://name.github.io直接访问

## 初始化博客

```
// 新建文件夹存储相关文件，并初始化
hexo init <folder>
// 进入文件夹
cd <folder>
// 安装依赖，这是个node.js命令
npm install
```

## 配置博客

初始化完成后，在建立的博客文件夹下会出现很多文件以及文件夹，其中**_config.yml**文件就是博客的配置文件，配置文件很长，但是大多数都不需要修改，需要修改的主要是以下几部分：

### 网站相关信息

```
# Site
title: InnoBlog
subtitle: May The Force Be With You
description: Innoecho's Blog
author: Innoecho
language: en
timezone: Asia/Shanghai
```

其中language和timezone都有相应的规范，需要查阅相关的规范文件来获得参数值

### 部署信息

```
# Deployment
deploy:
  type: git
  repo: git@github.com:Innoecho/Innoecho.github.io.git
  branch: master
```

这里的repo即为前面创建的Github仓库的地址

### 主题

```
# Extensions
theme: light
```

以我正在使用的主题为例，在创建的博客文件夹下执行
```
git clone git://github.com/tommy351/hexo-theme-light.git themes/light
```
并修改theme参数为light即可

到这里，Hexo博客框架就搭建完毕

# 使用博客

## 写博客

在博客目录下，执行
```
hexo new "文章标题"
```
即在博客目录/source/post文件夹下生成相应的文章标题.md文件

编辑该文件，输入文章内容

## 本地发布

写好博客之后，执行
```
hexo server
```
server命令将静态文件发布到本地，通过\*http://localhost:4000\*访问

## 部署

本地发布检查没问题之后，需要将静态文件部署到Github仓库，才可以让其他人通过互联网访问，执行
```
hexo generate
hexo deploy
```
其中，generate命令根据md文件以及相应的配置文件生成静态文件，deploy命令将生成的静态文件部署到Github的仓库中

现在，整个博客的发布流程到此结束

# TroubleShooting

## Deployer not found: git

执行
```
npm install hexo-deployer-git --save
```
然后进行部署

## 博客源文件的保存

`hexo deploy`命令是将生成的静态页面部署到Github仓库，如果想要同时保存源文件，那么可以在Github仓库中新建一个分支来保存，具体操作流程如下：

```
// git初始化，在博客目录下执行
git init
// 添加仓库地址
git remote add origin https://github.com/username/reponame.git
// 新建分支并切换到新建的分支
git checkout -b source
// 添加所有本地文件到git
git add .
// git提交
git commit -m ""
// 文件推送到source分支
git push origin source
```

在更新博客的源文件之后，除了使用hexo命令将静态页面部署到Github，还需要使用git命令将源文件push到Github仓库

```
// 添加源文件
git add .
// git提交
git commit -m ""
// push源文件
git push origin source
```

另外，由于public文件夹由hexo根据source文件夹以及配置文件生成，所以没有必要保存public文件夹。对此，可以在博客目录下建立**.gitignore**文件，将以下内容添加到文件中
```
node_modules/
public/
.deploy*/
```







