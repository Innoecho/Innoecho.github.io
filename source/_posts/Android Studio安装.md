---
title: Android Studio安装
date: 2017-03-02 10:23:00
tags: [Android]
categories: Android
---

Android Studio的安装方法
<!--more-->

# 下载

* [Android Studio](http://tools.android-studio.org/index.php)
* [JDK](http://www.oracle.com/technetwork/java/javase/downloads/index.html)

# 屏蔽向导

将下载好的Android Studio解压到想要放置的位置，修改`/bin/idea.properties`，在最末行添加`disable.android.first.run=true`，这样在初次运行时就可以避开向导。

# 设置代理

运行`/bin/studio.sh`，直接进入成功安装后的界面，但此时无法新建工程，SDK Manager也无法运行。在setting中的Proxy中设置代理，可用的代理在[这里](http://tools.android-studio.org/index.php/proxy)

# 安装依赖库

可能由于我的电脑是64位系统，如果直接安装会出现**unable to run mksdcard sdk tool**这样的错误，因此，要安装Android Studio还必须安装以下依赖库

`sudo apt-get install lib32z1 lib32ncurses5 lib32bz2-1.0 lib32stdc++6`

# 正常安装

恢复`/bin/idea.properties`，再次运行`/bin/studio.sh`，正常安装即可。

# 虚拟机安装

Android Studio自带的模拟器虽然在不断完善，然而我采用号称最快的Android模拟器——Genymotion
- 首先安装[Virtualbox](https://www.virtualbox.org/)，这个需要注意版本要在5.0.4以上
- 安装[Genymotion](https://www.genymotion.com/)，除了安装以外，还需要注册账号，并申请License
- 安装Androd Studio插件，如果网络不给力的话，可以在[这里](https://www.genymotion.com/plugins/)下载手动安装
- 这样就可以在Android Studio中启动Genymotion，第一次运行需要指定安装目录
- 下载想要的Android机，Enjoy~
