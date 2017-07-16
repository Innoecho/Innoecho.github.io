---
title: MathJax Test
date: 2017-07-14 09:40:23
tags: [Hexo, Math]
categories: Hexo
---

Use MathJax to show math equation
<!-- more -->

$$
\begin{eqnarray}
\nabla\cdot\vec{E} &=& \frac{\rho}{\epsilon_0} \\
\nabla\cdot\vec{B} &=& 0 \\
\nabla\times\vec{E} &=& -\frac{\partial B}{\partial t} \\
\nabla\times\vec{B} &=& \mu_0\left(\vec{J}+\epsilon_0\frac{\partial E}{\partial t} \right)
\end{eqnarray}
$$

## Install
MathJax allows you to include mathematics in your web pages, either using LaTeX, MathML, or AsciiMath notation, and the mathematics will be processed using JavaScript to produce HTML, SVG or MathML equations for viewing in any modern browser.

to use MathJax in the blog, We need to install a [plugin](https://github.com/hexojs/hexo-math) to auto-deploy the Mathjax:
```
npm install hexo-math --save
hexo math install
```
**don't add anything in _config.yml!!!**

## Note
if there are some trouble in using MathJax, Modify the file：**./node_modules/marked/lib/marked.js**:
* step1
replace
``escape: /^\\([\\`*{}\[\]()# +\-.!_>])/,``
with
``escape: /^\\([`*\[\]()# +\-.!_>])/,``
* step2
replace
``em: /^\b_((?:[^_]|__)+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,``
with
``em:/^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,``

**please ignore above if your MathJax work well**

## Use
Use LaTeX to write math equation,like:
1. sup(^) and sub(_)
$$ e^{i\pi}+1=0 $$
2. frac
$$ \frac{1}{3} $$
3. sqrt
$$ \sqrt[n]{x+y} $$
4. vector
$$ \vec{a}\cdot\vec{b} = 0 $$
5. integrate
$$ \int_0^1x^2{\rm d}x $$
6. limit
$$ \lim_{n \rightarrow + \infty}\frac{1}{n(n+1)} $$
7. sum
$$ \sum_{i=0}^n\frac{1}{i^2} $$
8. greek alphabet
$$
\alpha \quad \beta \quad \gamma \quad \delta \quad \epsilon \quad \theta \quad \lambda \quad
\mu \quad \eta \\
\nu \quad \xi \quad \pi \quad \rho \quad \sigma \quad \tau \quad \phi \quad \psi \quad \omega
$$


