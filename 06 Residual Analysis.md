# 06 - Residual Analysis and Model Adequacy

### 1. Final goal

Is there Gaussian distribution of random error term in model?

### 2. Previous Test on overfitting

#### 2.1. The $ R^2$

Coeffecient of determination: $R^2=\frac{SS_R}{SS_T}=1-\frac{SS_{Res}}{SS_T}$, where $0\leq R^2\leq 1$, the larger the better

#### 2.2. The $R^2_{adj}$

Adjusted $R^2$: $R^2_{adj}=1-\frac{SS_{Res}/(n-p)}{SS_T/(n-1)}$, so it regularize the number of coefficient to avoid overfitting problem.

### 3. Measure the residuals

$$
SS_T=\sum_{i=1}^n(y_i-\bar{y})^2\\
SS_R=\sum_{i=1}^n(\hat{y_i}-\bar{y})\\
SS_{Res}=\sum_{i=1}^n(y_i-\hat{y}_i)^2=SS_T-SS_R\\
e=y-\hat{y}=(I-X(X^TX)^{-1}X^T)y=(I-H)y=(I-H)\varepsilon\\
MS_{Res}=SS_{Res}/(n-p)
$$

#### 3.1. Hat matrix $H$ has property:

(1) idempotent 

(2) has all diagonal entry $0<h_{ii}<1$. Proof in **Appendix 1**. 

(3) For $y=\beta_0+\beta_1x+\varepsilon$, $h_{ii}=\frac{1}{n}+\frac{(x_i-\bar{x})^2}{S_xx}$. Proof in **Appendix 2**. 

(4) $Hy=\hat{H}y$ so $He=0$. Proof in **Appendix 3**. 

(5) Eigenvalues of H has #p ones and #n-p zeros. Proof in **03 Multiple Linear Regression - Appendix 2**

#### 3.2. Definition of residual

We want to assume $e\sim N(0,\sigma^2(I-H))$ where $var(e_i)=\sigma^2(1-h_{ii})$ and $cov(e_i,e_j)=-\sigma^2h_{ij}$. Next nodes will show how to prove the Gaussian Property

### 4. Scaling Residuals

#### 4.1. Standardized Residuals

$d_i=\frac{e_i}{\sqrt{MS_{Res}}}$ for data $i=1,2,...,n$, where it assumes $var(e_j)\approx MS_{Res}=\sigma^2$. Drawback: violates $var(e_i)=\sigma^2(1-h_{ii})$. Not t-distributed.

#### 4.2. Studentized Residuals

$r_i=\frac{e_i}{\sqrt{MS_{Res}(1-h_{ii})}}$ it can approximate as more data. However, it is affected by large $h_{ii}$ when size of data is small. Drawback 2: (分子、分母) are dependent. Not t-distributed.

#### 4.3. Standardized PRESS Residuals

The prediction error sum of squares (PRESS) is $e_{(i)}=y_i-\hat{y}_{(i)}$, just omit a data point + run regression + get $\beta$ + regress on oringinal X + find residual. So $\hat{y}_{(i)}=x_{i}^T\hat{\beta}_{(i),LS}=x_i^T(X_{(i)}^TX_{(i)})^{-1}X_{(i)}^Ty_{(i)}$. Denoting $X=\begin{bmatrix}x_1^T\\x_2^T\\\vdots\\x_n^T\end{bmatrix},x_i=\begin{bmatrix}1\\x_{i1}\\\vdots\\x_{ik^T}\end{bmatrix}$

Note, $e_{(i)}=\frac{e_i}{1-h_{ii}}$ (derivation in **Appendix 4**), so no need to do regression n times.

4.3.1. **PRESS residual** is $PRESS=\sum_{i=1}^n\large(\frac{e_i}{1-h_{ii}}\large)^2$ which measures how regression model preforms in predicting new data

4.3.2. **$R_{predict}^2$** = $1-\frac{PRESS}{SS_T}$ is the prediction capability, help indentify **outliers** ($R^2$ only test overfitting)

4.3.3. **Standardized PRESS Residual** = $\frac{e_i}{Var(e_{(i)})}=\frac{e_i}{\sqrt{\sigma^2(1-h_{ii})}}$, drawback: $e_i$ and $h_{ii}$ are dependent, $\sigma^2$ unknown. So, not t-distribution.

#### 4.4. R-Student Residuals/ Näive Residuals

Denote $S_{(i)}^2=\frac{\sum_{j\neq i}^n(y_j-x_{j}^T\hat{\beta}_{(i)})^2}{n-p-1}=\frac{(n-p)MS_{Res}-e_i^2/(1-h_{ii})}{n-p-1}\neq \sum_{j=1,j\neq i}^ne^2_{(i)}$

Externally studentized residual / R-student residual $t_i=\frac{e_i}{\sqrt{S_{(i)}^2(1-h_{ii})}}$ for $i=1,2,..,n$, so $t_i\sim t_{n-p-1}$.

To prove the distribution, (1) prove $\frac{S_{(i)}^2(n-p-1)}{\sigma^2}\sim \chi^2_{n-p-1}$ . The proof is in **Appendix 5** (2) prove $e_i$ and $S_{(i)}^2$ independent. The proof is in **Appendix 6** . (3) Then $e_i\sim N(0,\sigma^2(1-h_{ii}))$, so $t_i=[\frac{e_i}{\sqrt{\sigma^2(1-h_{ii})}}]/\sqrt{[\frac{S_{(i)}^2(n-p-1)}{\sigma^2}]/(n-p-1)}\sim t_{n-p-1}$

-----

### Appendix

#### Appendix 1

$h_{ii}=x_i^T(X^TX)^{-1}x_i$ ($x_i$ represents the column), where $HH=H$, so $h_{ii}=\sum h_{ij}^2=h_i^Th_i\geq0$

So $h_{ii}=h_{ii}^2+\sum_{i\neq j}^nh_{ij}^2\geq0$ for $H_{n\times n}$. Then $h_{ii}-h_{ii}^2=\sum_{i\neq j}^nh_{ij}^2\geq0$, we have $h_{ii}(1-h_{ii})\geq 0)$. Since $h_{ii}\geq0$, so $h_{ii}\leq 1$

#### Appendix 2

We have $y=[1,x][\beta_0;\beta_1]+\varepsilon$

So, denote $X=\begin{bmatrix}1&x_1\\1&x_2\\1&x_4\\\vdots&\vdots\\1&x_n\end{bmatrix}$, then $X^TX=\begin{bmatrix}N&\sum_{i=1}^nx_i\\\sum_{i=1}^nx_i&\sum_{i=1}^nx_i^2\end{bmatrix}$, so $(X^TX)^{-1}=\frac{1}{N\sum_{i=1}^nx_i^2-(\sum_{i=1}^nx_i)^2}\begin{bmatrix}\sum_{i=1}^nx_i^2&-\sum_{i=1}^nx_i\\-\sum_{i=1}^nx_i&N\end{bmatrix}$

So, $(X^TX)^{-1}X^T=\frac{1}{NS_{XX}}\begin{bmatrix}\sum_{i=1}^nx_i^2-x_1\sum_{i=1}^nx_i&|&\sum_{i=1}^nx_i^2-x_2\sum_{i=1}^nx_i&|&\cdots\\-\sum_{i=1}^nx_i+Nx_1&|&\sum_{i=1}^nx_i+Nx_2&|&\cdots\end{bmatrix}$

So, $h_{11}=\frac{N[x_1^2-2x_1\frac{\sum x_i}{N}+(\frac{\sum x_i}{N})^2+\frac{\sum x_i^2}{N}-(\frac{\sum x_i}{N})^2]}{NS_{xx}}=\frac{(x_1-\bar{x})^2}{S_{xx}}+\frac{1}{N}$

**Note**, as $n\rightarrow\infty$, $h_{ii}=0$

Recall $Var(x)=E[\sum(x_i-\bar{x})^2]=\sum x_i^2+n\bar{x}=\sum x_i^2+\frac{\sum x_i}{n}$

#### Appendix 3

$He=H(y-\hat{y})=H(I-H)y=X(X^TX)^{-1}X^T(I-X(X^TX)^{-1}X^T)y\\=[X(X^TX)^{-1}X^T-X(X^TX)^{-1}X^T]y=0y=0$

#### Appendix 4

*这部分证明一定要手算*

$e_{(i)}=y_i-x_i^T(X_{(i)}^TX_{(i)})^{-1}X_{(i)}^Ty_{(i)}$

(1) For term $[X^T_{(i)}X_{(i)}]_{m,n}=\sum_{k=1,k\neq i}^nX_{mk}X_{kn}=X^TX-x_ix_i^T=\text{   (a)}$ where $x_i$ denotes the i-th column of data.

That, is $x_ix_i^T=\begin{bmatrix}|&|&...&|\\\text{[}x_{i1}x_{i1}\text{]}_{1\times 1}&x_{i2}x_{i2}&...&x_{ip}x_{ip}\\|&|&...&|\end{bmatrix}$

Use **Theorem** of [Sherman-Morrison-Wood](https://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec12.pdf): $(A+UCV)^{-1}=A^{-1}-A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}$

Denote $A=[X^TX]_{p\times p}$, $U=[x_i]_{p\times 1}$, $C=[-1]_{1\times 1}$, $V=[x_i^T]_{1\times p}$

So 代入得 $\text{(a)}^{-1}=(X^TX)^{-1}-\frac{1}{h_{ii}-1}(X^TX)^{-1}x_ix_i^T(X^TX)^{-1}$ (use formula in appendix 1 for $h_{ii}$)

So, 代入得 $e_{(i)}=y_i-\frac{1}{1-h_{ii}}x_i^T(X^TX)^{-1}X^T_{(i)}y_{(i)}\text{ (b)}$

Note, for $X=\begin{bmatrix}x_1^T\\x_2^T\\\vdots\\x_n^T\end{bmatrix},x_i=\begin{bmatrix}1\\x_{i1}\\\vdots\\x_{ik^T}\end{bmatrix}$, $X^T_{(i)}y_{(i)}=X^Ty-x_{i}y_i$ for $[x_i]_{p \times 1}$ and $[y_i]_{1\times1}$

So $\text{ (b)}=\frac{y_i-x_i^T\hat{\beta}}{1-h_{ii}}=\frac{e_i}{1-h_{ii}}$

#### Appendix 5

$$
\begin{align}
S_{(i)}^2&=\frac{1}{n-p-1}[\sum_{j=1,j\neq i}^n(y_j-x_{j}^T\hat{\beta}_{(i)})^2]\\
&=\frac{1}{n-p-1}(y_{(i)}-X_{(i)}(X^T_{(i)}X_{(i)})^{-1}X^T_{(i)}y_{(i)})^T(y_{(i)}-X_{(i)}(X^T_{(i)}X_{(i)})^{-1}X^T_{(i)}y_{(i)})\\
&=\frac{1}{n-p-1}y_{(i)}^T(I-H_{(i)})y_{(i)}
\end{align}
$$

We know $y_{(i)}\sim N(X_{(i)}\beta,\sigma^2I)$

Recall theorem 2 in section 2.4 in 03 Multiple Linear Regression: $p'=rank(I-H_{(i)})=n-p-1$, $\lambda'=E[y_{(i)}]^T(I-H_{(i)})E[y_{(i)}]/\sigma^2=0/\sigma^2=0$, So $y_{(i)}^T(I-H_{(i)})y_{(i)}/\sigma^2\sim \chi^2_{n-p-1,0}$. So $(n-p-1)S_{(i)}^2/\sigma^2\sim\chi^2_{n-p-1}$

#### Appendix 6

prove $e_i$ and $S_{(i)}^2$ independent. we prove $e_{(i)}$ and $S_{(i)}^2$ independent

We have $[e_i]_{1\times1}=y_i-\hat{y}_{(i)}=y_i-x_i^T(X_{(i)}^TX_{(i)})^{-1}X_{(i)}^Ty_{(i)}$, and $S_{(i)}^2=\frac{1}{n-p-1}y_{(i)}^T[I-X_{(i)}(X_{(i)}^TX_{(i)})^{-1}X_{(i)}^T]y_{(i)}$

Recall theorem 4 in appendix 5 in 03 Multiple Linear Regression: (1) scaler $y_i$ and leftover vector $y_{(i)}$ are definitely independent (2) $B=x_i^T(X_{(i)}^TX_{(i)})^{-1}X_{(i)}^T$, $A=[I-X_{(i)}(X_{(i)}^TX_{(i)})^{-1}X_{(i)}^T]$, $\Sigma=\sigma^2I$, we have $B\Sigma A=0$

So  $e_{(i)}$ and $S_{(i)}^2$ independent. Since $h_{ii}$ is just a scaler, so $e_i$ and $S_{(i)}^2$ independent