---
title: GDB_Test
date: 2017-07-27 17:13:36
tags: [GDB, Ubuntu, Debug]
categories: C++
---

使用GDB工具在Ubuntu下调试C++程序

GDB是GNU开源组织发布的一个强大的UNIX下的程序调试工具。不同于其他图形界面的调试工具，GDB是运行在命令行的调试工具，可能在使用上会有些不适应，但是功能绝对完备，甚至更强大。目前我在Ubuntu下的工作方式是使用Sublime写代码，用GDB调试代码，因此，在此列出GDB调试代码的常用命令。
<!--more-->

# 实例

下面给出一个简单的例子来说明如何使用gdb调试程序

## 源程序

```c++
#include <iostream>
#include <vector>

int printvector(std::vector<int> nums)
{
	static int callCount = 0;

	callCount++;
	for (int i = 0; i < nums.size(); ++i)
		std::cout << nums[i] << "  ";
	std::cout << std::endl;

	return callCount;
}


int main()
{
	std::vector<int> nums{1, 2, 3, 4, 5};

	std::cout << printvector(nums) << std::endl;

	return 0;
}
```

将上面内容保存为gdb_test.cpp，并编译生成可执行文件

```sh
g++ -g -std=c++11 gdb_test.cpp -o gdb_test
```

需要注意的是，编译的时候应该加上参数**-g**，把源代码信息同时编译到可执行文件中，便于调试，否则就只能面对汇编进行调试了……

## 调试

### 启动gdb并读入待调试程序

命令行中输入`gdb`并执行，即可进入gdb调试环境，`file gdb_test`读入待调试程序，读入成功后，输入命令`list|l`，gdb将会显示源代码，每次10行，按回车继续输出（gdb中回车表示再次执行上次命令）

```sh
inno@Inno-pc:~/Source/Cpp_Tutorial/GDBTest$ gdb             <----gdb启动调试环境
GNU gdb (Ubuntu 7.11.1-0ubuntu1~16.04) 7.11.1
Copyright (C) 2016 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
<http://www.gnu.org/software/gdb/documentation/>.
For help, type "help".
Type "apropos word" to search for commands related to "word".
(gdb) file gdb_test                                           <----读入待调试程序
Reading symbols from gdb_test...done.
(gdb) l                                                          <----显示源代码
1	#include <iostream>
2	#include <vector>
3	
4	void printvector(std::vector<int> nums)
5	{
6		static int callCount = 0;
7	
8		callCount++;
9		for (int i = 0; i < nums.size(); ++i)
10			std::cout << nums[i] << "  ";
(gdb)                                                     <----回车表示重复执行命令
11		std::cout << std::endl;
12	}
13	
14	int main()
15	{
16		std::vector<int> nums{1, 2, 3, 4, 5};
17		printvector(nums);
18		
19		return 0;
20	}(gdb)
```

### 调试程序

`run|r`命令运行程序，由于此时没有进行任何设置，所以程序会一路跑完，最后输出结果

```sh
(gdb) r                                                          <----运行程序
Starting program: /home/inno/Source/Cpp_Tutorial/GDBTest/gdb_test 
1  2  3  4  5  
1
[Inferior 1 (process 11520) exited normally]
```

调试程序一个最重要的工作就是添加断点，gdb提供了多种添加断点的方式：

1. 行号：break|b 行号
2. 函数：break|b 函数名
3. 条件断点： break|b 行号（函数名） if 条件

```sh
(gdb) b 16                                                 <----在第16行添加代码
Breakpoint 1 at 0x400c2c: file gdb_test.cpp, line 16.
(gdb) b printvector                              <----在函数printvector处添加代码
Breakpoint 2 at 0x400b43: file gdb_test.cpp, line 8.
```

添加完断点后，运行程序，可以看到程序停在了第一个断点处

```sh
(gdb) r
Starting program: /home/inno/Source/Cpp_Tutorial/GDBTest/gdb_test 

Breakpoint 1, main () at gdb_test.cpp:16
16		std::vector<int> nums{1, 2, 3, 4, 5};
```

使用`step|s`进行单步跟踪，如果有函数调用则进入函数，也可以使用`next|n`进行单步跟踪，不同的是，这个命令不会进入函数调用。




使用`print|p 变量名`可以查看变量

```sh
(gdb) p nums
$1 = std::vector of length 5, capacity 5 = {1, 2, 3, 4, 5}
```