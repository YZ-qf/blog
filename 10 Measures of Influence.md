#! https://zhuanlan.zhihu.com/p/374101523
# 10 Measures of Influence

The nonlinear models are not covered in these notes, like Neural Networ, Logistic Regression, because I have written machine learning notes on those. Newton's Method, Gauss-Newton's method will not be covered, so please see optimization notes for details.

### Leverage score

Recall 06 Residual Analysis, $\hat{y}=Hy$, thus $h_{ii}$ is leverage score, or the weight of $y_i$ is predicting $\hat{y}_i$

##### High leverage point

==**if $h_{ii}\geq 2p/n$, i-th point is high-leverage point,**== for $X_{n\times p}$ in $y=X\beta$

### Influencial point

Influencial point > outliers. if Influencial point are on the trend of data, but x value is far away from majority, the point is fine, not outlier.

### Detect Single Influencial Point

to quantify the influence in multiple linear regression model, note, $e_i$ and $h_{ii}$ both matters, so consider $g(e_i,h_{ii})$

#### Cook's Measure

We have $\hat{\beta}$ from LS, and $\hat{\beta}_{(i)}$ from LS with i-th data deleted

For a single influencial point, $D_i=\frac{(\hat{\beta}-\hat{\beta}_{(i)})^TM(\hat{\beta}-\hat{\beta}_{(i)})}{c}=\frac{r_i^2}{p}\frac{h_{ii}}{1-h_{ii}}$ (derivation see **Appendix 1**), $X_{n\times p}$, $M=X^TX$, $c=p\cdot MS_{Res}$, $D_i\sim F_{p,n-p}$ (proof see **Appendix 2**), $r_i$ is studentdized residual. ==**$D_i\geq1$ is high influential point**==, $F_{0.5,n,n-p}\approx 1$

#### DFFITS Measure (optional)

$DFFITS_i=\frac{\hat{y}_i-\hat{y}_{(i)}}{\sqrt{S_{(i)}^2h_{ii}}}=[\sqrt{\frac{h_{ii}}{1-h_{ii}}}]t_i$ [proof see **Appendix 3**], where $var(\hat{y}_i)=\sigma^2h_{ii}$, $t_i$ is R-student residual.==**$DFFITS_i^2>4p/n$ maybe high influencial point.**==

### Detect Group of Influencial Point

#### Cook's Measure

$D_i=\frac{(\hat{\beta}-\hat{\beta}_{(i)})^TM(\hat{\beta}-\hat{\beta}_{(i)})}{c}$ for a set of point $\{x_{i}\}_{i\in I}$ are deleted.

### Deal with Influencial Point

So, after we find all Influencial Point, we can either discard them all, or we can use Robust Estimation as in the next lecture.

----

### Appendix

#### Appendix 1

For $y=X\beta+\varepsilon$,
$$
\begin{aligned}
D_i&=\frac{(\hat{\beta}-\hat{\beta}_{(i)})^TX^TX(\hat{\beta}-\hat{\beta}_{(i)})}{p\cdot MS_{Res}}\\
&=\frac{(\hat{y}_{(i)}-\hat{y})^T(\hat{y}_{(i)}-\hat{y})}{p\cdot MS_{Res}}
\end{aligned}
$$
Recall [06 - Residual Analysis and Model Adequacy](https://zhuanlan.zhihu.com/p/359935375) Appendix 4, $X_{(i)}$ for i-th row deleted
$$
(X_{(i)}^TX_{(i)})^{-1}=(X^TX)^{-1}+\frac{(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}}{1-h_{ii}}
\tag{2}
$$
We have $x_{(i)}^Ty_{(i)}=X^Ty-x_iy_i$, multiply both side of (2) by it, we have,
$$
\hat{\beta}_{(i)}=\hat{\beta}-(X^TX)^{-1}x_iy_i+\frac{(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}(X^Ty-x_iy_i)}{1-h_{ii}}
$$
Consider scalar $y_i$ can be moved anywhere, from (2)
$$
\begin{aligned}
&(X^TX)^{-1}x_iy_i(1-h_{ii})-(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}(X^Ty-x_iy_i)\\
&=(X^TX)^{-1}x_iy_i-(X^TX)^{-1}x_ih_{ii}y_i\\&-(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}X^Ty+(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}x_iy_i\\
&=(X^TX)^{-1}x_iy_i-(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}X^Ty\\
&=(X^TX)^{-1}x_iy_i-(X^TX)^{-1}x_ix_i^T\hat{\beta}\\
&=(X^TX)^{-1}x_ie_i
\end{aligned}
$$
Thus, $\hat{\beta}-\hat{\beta}_{(i)}=\frac{(X^TX)^{-1}x_ie_i}{1-h_{ii}}\tag{3}$

Thus, 
$$
\begin{aligned}
D_i&=\frac{e_ih_{ii}e_i}{(1-h_{ii})^2p\cdot MS_{Res}}\\
&=\large(\frac{e_i}{\sqrt{MS_{Res}(1-h_{ii})}}\large)^2\frac{1}{p}\frac{h_{ii}}{1-h_{ii}}\\
&=r_i^2\frac{1}{p}\frac{h_{ii}}{1-h_{ii}}
\end{aligned}
$$
$r_i$ is the studendized residual.

#### Appendix 2

Thus, look at it, if we assume $\varepsilon\sim N(0,\sigma^2I)$, then $D_i$ is F distribution, [proof](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3867306/), if $D_i$ too large, reject $H_0$, say $i$-th data is influential, maybe outlier.

#### Appendix 3

$$
\begin{aligned}
DFFITS_i&=\frac{\hat{y}_i-\hat{y}_{(i)}}{\sqrt{S_{(i)}^2h_{ii}}}\\
&=\frac{x_i^T(\hat{\beta}-\hat{\beta}_{(i)})}{\sqrt{S_{(i)}^2h_{ii}}}\\
&=\frac{h_{ii}e_i}{(1-h_{ii})\sqrt{S_{(i)}^2h_{ii}}},\text{ by (3) in Appendix 1}\\
&=t_i\sqrt{\frac{h_{ii}}{1-h_{ii}}}
\end{aligned}
$$

