---
title: Hexo引用本地图片
date: 2017-05-07 16:32:21
tags: [Hexo]
categories: Hexo
---

通过使用[hexo-asset-image](https://github.com/CodeFalling/hexo-asset-image)插件引用本地图片，解决Hexo直接引用本地图片路径混乱的问题
<!-- more -->

## 配置

将*_config.yml*中参数*post_asset_folder*的值配置为*true*

这个参数使得在建立文件时，自动生成一个与文件同名的文件夹，用于存储资源文件

## 安装插件

在hexo目录下执行`npm install hexo-asset-image --save`命令安装插件

## 使用

假设在相应的资源文件下存在图片文件为*logo.png*，使用如下语法引用图片

`![logo](logo.png)`

![logo](logo.png)


