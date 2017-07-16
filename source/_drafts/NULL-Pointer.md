---
title: NULL_Pointer
date: 2017-07-14 17:25:49
tags:
categories:
---


<!--more-->
```c++
char word[11];
std::cin >> word;
std::cout << word << std::endl;
```

```c++
char* word;
std::cin >> word;
std::cout << word << std::endl;