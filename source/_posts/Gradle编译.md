---
title: Gradle编译
date: 2017-03-02 10:25:56
tags: [Gradle]
categories: Android
---

Android开发工具Gradle的安装以及使用
<!--more-->

# 下载Gradle

切换到项目的根目录，执行

`./gradlew -v`

查看所使用的Gradle的版本，如果是第一次执行，则会下载所需版本。

# 下载依赖项

执行

`./gradlew clean`

下载所需依赖项，这步遇到较多问题：

- **SDK Location not found**：执行`export ANDROID_HOME=<sdk dir>`配置环境变量
- **failed to find Build Tools revision**：修改build.gradle文件

我编译的是DroneKit的例子Hello Drone，这个例子教程中导入的库是**'com.o3dr.android:dronekit-android:2.3.+'**，github上的版本导入的库是**'com.o3dr.android:dronekit-android:2.7.+'**，然而这两种都会提示无法找到对应的库，在[jcenter](https://jcenter.bintray.com/com/o3dr/)中最新的库是**'com.o3dr.android:dronekit-android:2.9.0'**，导入这个库就不会报错了。

# 编译APK

执行

`./gradlew build`

编译生成APK文件
