# 05 - Polynomial regression

### Function

the hierarchical k-th order polynomial regression:

$y=\sum_{j=0}^k\beta_jx^j+\varepsilon$

### Overfitting

for $(y-X\beta)^T(y-X\beta)=0$, and $\hat{\beta}=X^{-1}y$ perfect fitting

### Forward selection

(1) Forward selection: Add parameters to increase order until t test for high order term is insignificant (t-test fails).

(2) Set #m parameters and use regularization

(3) Extrapolation (predict data) and interpolation (fill within data)

### Spline

### Problem for high order

1. Overfitting

##### 2. Multicolinearearity among X

1. essential multicollinearity: $\sum_{i=}1^pt_ix_i=0$ where $t_i$ exists

2. non-essential multicollinearity: for $x_1,x_2$ scaled to unit length, denote $r_{12},r_{1y},r_{2y}$ as correlation between $x_1\sim x_2$, $x_1\sim y$, and $x_2\sim y$. Then, $\hat{\beta}_1=\frac{r_{1y}-r_{12}r_{2y}}{1-r_{12}^2}$, $\hat{\beta_2}=\frac{r_{2y}-r_{12}r_{1y}}{1-r_{12}^2}$

   (a) absolute value of correlation between two variables $|r_{12}|$ are close to 1. (2) $var(\hat{\beta}_j) \rightarrow \infty$ as correlation approaches to 1. Proof see **Appendix 1**

##### Solve multicollinearity

1. solve partial: check correlation matrix to find linear-dependence between pairs of inputs

2. for eigenvalues of **unit-length scaled** $X^TX$, find **condition number** $\kappa=\frac{\lambda_{max}}{\lambda_{min}}$, if $\kappa>1000$, whole model is multi-collinear. **condition indices** are $\kappa_j=\frac{\lambda_\max}{\lambda_j}$ for j=1,2,...,p

3. To examine the collinearity, select eigenvector with smallest eigenvalue. Assume there is collinearity, then $\sum a_ix_i=0$, then replace a with eigenvector,  then omit $x_i$ with eigen vector $\rightarrow$ 0, for the rest $x_i$, we can express $x_i$ as linear combination of the rest x. 【E.g.】 so the elements of the eigenvector $t_6$ are the coefficients of the regressors in Eq. (9.1). This implies that

   $−0.44768x_1 −0.42114x_2 −0.54169x_3 −0.57337x_4 −0.00605x_5 −0.00217x_6 =0$ so $x_1 􏰀 −0.941x_2 − 1.120x_3 − 1.281x_4$

--------------

### Appendix

#### Appendix 1

(1) For $y=\beta_1x_1+\beta_2x_2+\varepsilon$, where $x_1,x_2,y$ are under unit length scaling of oringinal data (so mean=0, var=$r_{11}$=$r_{22}$=1)

So, $X^Ty=\begin{bmatrix}r_{1y}\\r_{2y}\end{bmatrix}$, $X^TX=\begin{bmatrix}r_{11}&r_{12}\\r_{21}&r_{22}\end{bmatrix}$, $(X^TX)^{-1}=\frac{1}{1-r_{12}^2}\begin{bmatrix}r_{22}&-r_{12}\\-r_{12}&r_{11}\end{bmatrix}=\frac{1}{1-r_{12}^2}\begin{bmatrix}1&-r_{12}\\-r_{12}&1\end{bmatrix}$

So $\hat{\beta}=(X^TX)^{-1}X^Ty=r_{xx}^{-1}r_{xy}$. So we can find $\hat{\beta}_1,\hat{\beta}_2$

(2) Also, $Cov(\hat{\beta})=\sigma^2(X^TX)^{-1}$, so $Var(\hat{\beta})=\sigma^2/(1-r_{12}^2)$ as $r_{12}$ approaches to 0, variance of beta approaches to infinity. If $x_1,x_2$ are independent, then $Var(\hat{\beta})=\sigma^2$

