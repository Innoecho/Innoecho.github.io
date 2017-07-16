---
title: Numpy Quickstart Tutorial
date: 2017-05-22 20:58:13
tags: [Python, Numpy]
categories: Python
---

QuickStart For Numpy
<!-- more -->

# 1. The Basics


```python
import numpy as np
# create am array and show
a = np.arange(15).reshape(3,5)
a
```




    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])




```python
# show every axis's length of an array
a.shape
```




    (3, 5)




```python
# show an array's dim
a.ndim
```




    2




```python
# show data type of an array
a.dtype
```




    dtype('int64')




```python
# show itemsize of an array
a.size
```




    15




```python
# show type of a
type(a)
```




    numpy.ndarray




```python
# create another array
b = np.array([6, 7, 8])
b
```




    array([6, 7, 8])



# 2. Array Creation
## 2.1 create array by regular Python list or tuple


```python
import numpy as np

a = np.array((2, 3, 4))
a
```




    array([2, 3, 4])




```python
b = np.array([5, 6, 7])
b
```




    array([5, 6, 7])




```python
c = np.array([[1, 2, 3], [4, 5, 6]])
c
```




    array([[1, 2, 3],
           [4, 5, 6]])



## 2.2 create array by functions


```python
# zeros, default type is float64
a = np.zeros((3,4))
a.dtype
```




    dtype('float64')




```python
# ones
b = np.ones((2, 3), dtype = np.int16)
b
```




    array([[1, 1, 1],
           [1, 1, 1]], dtype=int16)




```python
# empty, it create an array by random item

c = np.empty((2, 2))
c
```




    array([[  6.94428861e-310,   1.15911595e-316],
           [  4.76862566e+180,   1.63041663e-322]])




```python
# arange, give the begin, the step and the and

np.arange(10, 30, 5)
```




    array([10, 15, 20, 25])




```python
# linspace, give the begin, the end and the number of elements

np.linspace(0, 2, 9)
```




    array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])



# 3. printing Arrays


```python
# 1d array

a = np.arange(6)
print(a)
```

    [0 1 2 3 4 5]



```python
# 2d array

b = np.arange(12).reshape(4,3)
print(b)
```

    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]



```python
# too long array

print(np.arange(10000))
```

    [   0    1    2 ..., 9997 9998 9999]



```python
print(np.arange(10000).reshape(100, 100))
```

    [[   0    1    2 ...,   97   98   99]
     [ 100  101  102 ...,  197  198  199]
     [ 200  201  202 ...,  297  298  299]
     ..., 
     [9700 9701 9702 ..., 9797 9798 9799]
     [9800 9801 9802 ..., 9897 9898 9899]
     [9900 9901 9902 ..., 9997 9998 9999]]


# 4. Basic Operations
## 4.1 Arithmetic operations on arrays apply elementwise


```python
# + - * /
a = np.arange(4, dtype = np.float64)
b = np.array([2, 3, 4, 5])
print(a+b)
print(a-b)
print(a*b)
print(a/b)
```

    [ 2.  4.  6.  8.]
    [-2. -2. -2. -2.]
    [  0.   3.   8.  15.]
    [ 0.          0.33333333  0.5         0.6       ]



```python
# others
print(b**2)
print(np.sin(a))
print(a<3)
```

    [ 4  9 16 25]
    [ 0.          0.84147098  0.90929743  0.14112001]
    [ True  True  True False]


## 4.2 matrix product


```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[4, 2], [3, 1]])
# multi by element
print(A*B)
# matrix multi
print(A.dot(B))
```

    [[4 4]
     [9 4]]
    [[10  4]
     [24 10]]


## 4.3 unary operations


```python
a = np.arange(1, 5)
print(a)
print(a.sum())
print(a.max())
print(a.min())
```

    [1 2 3 4]
    10
    4
    1



```python
# for 2d

b = np.arange(1, 5).reshape(2, 2)
print(b)
print(b.sum())
print(b.sum(axis = 0))
```

    [[1 2]
     [3 4]]
    10
    [4 6]


# 5. Universal Functions


```python
B = np.arange(3)
print(np.exp(B))
print(np.sqrt(B))
```

    [ 1.          2.71828183  7.3890561 ]
    [ 0.          1.          1.41421356]


See also:

all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where

# 6. Indexing, Slicing and Iterating


```python
a = np.arange(10)**3
a
```




    array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])




```python
print(a[2])
print(a[2:5])
```

    8
    [ 8 27 64]



```python
a[:6:2] = -1000
print(a)
print(a[::-1])
```

    [-1000     1 -1000    27 -1000   125   216   343   512   729]
    [  729   512   343   216   125 -1000    27 -1000     1 -1000]



```python
# create arrays from function

def f(x, y):
    return 10*x+y

b = np.fromfunction(f, (5, 4), dtype = int)
print(b)
```

    [[ 0  1  2  3]
     [10 11 12 13]
     [20 21 22 23]
     [30 31 32 33]
     [40 41 42 43]]



```python
print(b[2,3])
print(b[0:5,1])
print(b[:,1])
print(b[-1])
```

    23
    [ 1 11 21 31 41]
    [ 1 11 21 31 41]
    [40 41 42 43]



```python
# flat is an iterator and flatten is a function
A = np.array([[1,2],[3,4]])
print(A)
print(A.flatten())
for element in A.flat:
    print(element)
```

    [[1 2]
     [3 4]]
    [1 2 3 4]
    1
    2
    3
    4


# 7. Shape Manipulation
## 7.1 Changing the shape of an array


```python
# create an array

a = np.floor(10*np.random.random((3,4)))
a
```




    array([[ 7.,  1.,  9.,  6.],
           [ 9.,  1.,  5.,  9.],
           [ 9.,  7.,  9.,  1.]])




```python
a.shape
```




    (3, 4)




```python
# return the array, flattened
print(a.ravel())
```

    [ 7.  1.  9.  6.  9.  1.  5.  9.  9.  7.  9.  1.]



```python
# return the array with modified shape
print(a.reshape(6,2))
```

    [[ 7.  1.]
     [ 9.  6.]
     [ 9.  1.]
     [ 5.  9.]
     [ 9.  7.]
     [ 9.  1.]]



```python
# return the array, transposed
print(a.T)
```

    [[ 7.  9.  9.]
     [ 1.  1.  7.]
     [ 9.  5.  9.]
     [ 6.  9.  1.]]



```python
# the above three commands all return a modified array, but do not change the original array
# the resize function modifies the array itself
print(a)
print(a.resize(2,6))
print(a)
      
```

    [[ 7.  1.  9.  6.]
     [ 9.  1.  5.  9.]
     [ 9.  7.  9.  1.]]
    None
    [[ 7.  1.  9.  6.  9.  1.]
     [ 5.  9.  9.  7.  9.  1.]]


## 7.2 Stacking together different arrays
vstack() and hstack()


```python
a = np.arange(1,5).reshape(2,2)
b = np.arange(5,9).reshape(2,2)
print(a)
print(b)
print(np.vstack((a,b)))
print(np.hstack((a,b)))
```

    [[1 2]
     [3 4]]
    [[5 6]
     [7 8]]
    [[1 2]
     [3 4]
     [5 6]
     [7 8]]
    [[1 2 5 6]
     [3 4 7 8]]



```python
np.r_[1:4,0,4]
```




    array([1, 2, 3, 0, 4])




```python
np.c_[1,3,5,7]
```




    array([[1, 3, 5, 7]])



## 7.3 Splitting one array into several smaller ones


```python
a = np.arange(1,25).reshape(2,12)
print(a)
```

    [[ 1  2  3  4  5  6  7  8  9 10 11 12]
     [13 14 15 16 17 18 19 20 21 22 23 24]]



```python
print(np.hsplit(a,3))
```

    [array([[ 1,  2,  3,  4],
           [13, 14, 15, 16]]), array([[ 5,  6,  7,  8],
           [17, 18, 19, 20]]), array([[ 9, 10, 11, 12],
           [21, 22, 23, 24]])]



```python
# split the array after the third and the fourth column

print(np.hsplit(a,(3,4)))
```

    [array([[ 1,  2,  3],
           [13, 14, 15]]), array([[ 4],
           [16]]), array([[ 5,  6,  7,  8,  9, 10, 11, 12],
           [17, 18, 19, 20, 21, 22, 23, 24]])]


# 8. Copies and Views
## 8.1 No Copy at All


```python
# Simple assignments

a = np.arange(12)
b = a
print(b is a)
b.shape = 3,4
print(a.shape)
```

    True
    (3, 4)


## 8.2 View or Shallow Copy


```python
# the view method creates a new array object that share the same data

c = a.view()
print(c is a)
print(c.base is a)
print(c.flags.owndata)
```

    False
    True
    False



```python
# a's shape doesn't change

c.shape = 2,6
print(a.shape)
```

    (3, 4)



```python
# a's data changes
print(c)
c[0,4] = 1234
print(c)
print(a)
```

    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]]
    [[   0    1    2    3 1234    5]
     [   6    7    8    9   10   11]]
    [[   0    1    2    3]
     [1234    5    6    7]
     [   8    9   10   11]]



```python
# slicing an array returns a view of it

s = a[:,1:3]
s[:] = 10
print(a)
```

    [[   0   10   10    3]
     [1234   10   10    7]
     [   8   10   10   11]]


## 8.3 Deep Copy


```python
# the copy method makes a complete copy of the array and its data

d = a.copy()
print(d is a)
print(d.base is a)
d[0,0] = 9999
print(a)
```

    False
    False
    [[   0   10   10    3]
     [1234   10   10    7]
     [   8   10   10   11]]


# 9. Indexing with Boolean Arrays


```python
import numpy as np
import matplotlib.pyplot as plt
def mandelbrot(h, w, maxit=20):
    """ return an image of the Mandelbrot fractal of size (h, w)"""
    y,x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y * 1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype = int)
    
    for i in range(maxit):
        z = z**2 + c
        # who is diverging
        diverge = z*np.conj(z) > 2**2
        # who is diverging now
        div_now = diverge & (divtime == maxit)
        # note when
        divtime[div_now] = i
        # avoid diverging too much
        z[diverge] = 2
        
    return divtime

plt.imshow(mandelbrot(400,400))
plt.show()
```


![](output_66_0.png)
