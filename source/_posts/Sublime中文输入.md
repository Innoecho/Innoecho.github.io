---
title: Sublime中文输入
tags: Sublime
categories: Sublime
date: 2017-03-02 16:24:01
---

解决Sublime中无法输入中文的问题
<!--more-->

### 准备工作

安装Sublime与搜狗输入法

### 生成所需文件

- 将下面代码保存到~目录下的sublime_imfix.c中

```c
#include <gtk/gtkimcontext.h>
	void gtk_im_context_set_client_window (GtkIMContext *context,
	          GdkWindow    *window)
	{
	  GtkIMContextClass *klass;
	  g_return_if_fail (GTK_IS_IM_CONTEXT (context));
	  klass = GTK_IM_CONTEXT_GET_CLASS (context);
	  if (klass->set_client_window)
	    klass->set_client_window (context, window);
	  g_object_set_data(G_OBJECT(context),"window",window);
	  if(!GDK_IS_WINDOW (window))
	    return;
	  int width = gdk_window_get_width(window);
	  int height = gdk_window_get_height(window);
	  if(width != 0 && height !=0)
	    gtk_im_context_focus_in(context);
	}
```

- 将上面的C文件编译成共享库libsublime-imfix.so

```bash
gcc -shared -o libsublime-imfix.so sublime_imfix.c  `pkg-config --libs --cflags gtk+-2.0` -fPIC
```

- 将libsublime-imfix.so移动到Sublime文件夹中

```bash
sudo mv libsublime-imfix.so /opt/sublime_text/
```

### 修改文件/usr/bin/subl

执行

```bash
sudo gedit /usr/bin/subl
```

将

```bash
#!/bin/sh
exec /opt/sublime_text/sublime_text "$@"
```

修改为

```bash
#!/bin/sh
LD_PRELOAD=/opt/sublime_text/libsublime-imfix.so exec /opt/sublime_text/sublime_text "$@"
```

此时，在命令中执行**subl**将可以使用搜狗输入法输入中文

### 修改文件sublime_text.desktop

```bash
sudo gedit /usr/share/applications/sublime_text.desktop
```

将[Desktop Entry]中的字符串

```bash
Exec=/opt/sublime_text/sublime_text %F
```

修改为

```bash
Exec=bash -c "LD_PRELOAD=/opt/sublime_text/libsublime-imfix.so exec /opt/sublime_text/sublime_text %F"
```

将[Desktop Action Window]中的字符串

```bash
Exec=/opt/sublime_text/sublime_text -n
```

修改为

```bash
Exec=bash -c "LD_PRELOAD=/opt/sublime_text/libsublime-imfix.so exec /opt/sublime_text/sublime_text -n"
```

将[Desktop Action Document]中的字符串

```bash
Exec=/opt/sublime_text/sublime_text --command new_file
```

修改为

```bash
Exec=bash -c "LD_PRELOAD=/opt/sublime_text/libsublime-imfix.so exec /opt/sublime_text/sublime_text --command new_file"
```
