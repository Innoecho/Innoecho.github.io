---
title: Apt_get Update TroubleShooting
date: 2017-03-02 10:17:15
tags: Apt-get
categories: Ubuntu
---

使用Apt-get的问题合集
<!--more-->

# 公钥

**W: 以下 ID 的密钥没有可用的公钥：**
**8B48AD6246925553**

执行

`gpg --keyserver subkeys.pgp.net --recv xxx`

`gpg --export --armor xxx | sudo apt-key add -`

xxx为密钥后8位

# list

**无法下载 http://dl.google.com/linux/chrome/deb/dists/stable/Release  Unable to find expected entry 'main/binary-i386/Packages' in Release file (Wrong sources.list entry or malformed file)**

- `cd /etc/apt/sources.list.d`
- `sudo gedit google-chrome.list`
- 将旧的源列表修改为**deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main**
- 原因：官方的Chrome不再提供32位包
