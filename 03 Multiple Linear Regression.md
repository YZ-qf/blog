# Multiple Linear Regression

#### 1. Polynomial model

$$
y=\beta_0+\beta_1x_1+\beta_1x_2+...\beta_kx_k+\varepsilon=X\boldsymbol{\beta}+\varepsilon
$$

, where (1) $\beta_j $ are called regression coefficients. 

(2) For the least-squares estimation, we often assume $\varepsilon$ has zero mean and unknown variance $\sigma^2$. For the maximum-likelihood estimation, we need to assume full knowledge/distribution of $p(\epsilon)$.

(3) Interaction model can be re-written as polynomial model

(4) We have
$$
\boldsymbol{y}=\begin{bmatrix}
y_1\\y_2\\\vdots\\y_n
\end{bmatrix}_{n\times1}
,\text{ }\boldsymbol{X}=\begin{bmatrix}
1&x_{1,1}&x_{1,2}&\cdots&x_{1,k}\\
1&x_{2,1}&x_{2,2}&\cdots&x_{2,k}\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&x_{n,1}&x_{n,2}&\cdots&x_{n,k}\\
\end{bmatrix}_{n\times p}\\
\text{ }\boldsymbol{\beta}=\begin{bmatrix}
\beta_0\\\beta_1\\\vdots\\\beta_k
\end{bmatrix}_{p\times1},\text{ }\boldsymbol{\varepsilon}=\begin{bmatrix}
\varepsilon_1\\\varepsilon_2\\\vdots\\\varepsilon_n
\end{bmatrix}_{n\times1}
$$
, where $p=k+1$, and $\varepsilon_1, \varepsilon_2 ,...,\varepsilon_n$ are mutually uncorrelated (no need to assume iid)

----

#### 2. LS estimation

**2.1. Optimization**
$$
\hat{\beta}=\arg\min_\beta S(\beta)=\arg\min_\beta(y-X\beta)^T(y-X\beta)\\=(X^TX)^{-1}X^Ty
$$
, where **$(X^TX)^{-1}$ exists**. The matrix derivative is shown in the Appendix 1.

**2.2. Estimation for $\sigma^2$**
$$
\begin{align}
e&=y-\hat{y}=y-X(X^TX)^{-1}X^Ty=(I-H)y\\
SS_{Res}&=\sum_{i=1}^ne_i^2=e^Te\\
\hat{\sigma}^2&=\frac{SS_{Res}}{n-p}=MS_{Res}
\end{align}
$$
, where $H_{n\times n}$ and $I-H$ are **idempotent matrix**: (1) Symmetric (2) $H\cdot H=H^2=H$ (3) $rank(H)=tr(H)$ (4) $I-H$ is idempotent, but $H-I$ is not. (5) Eigen value is either 0 or 1 (so (3) holds), see prove in Appendix

**2.3. Unbiasness of $\sigma^2$**

(1) Definition: Assume there is no model mismatch, that is $\hat{y}=X\hat{\beta}$ is the true one. Then $MS_{Res}$ is the unbiased estimator of $\hat{\sigma}^2$, that is $\hat{\sigma}^2=\frac{E(SS_{Res})}{n-p}=\sigma^2$

(2) We need **Theorem 1**: for constant $A_{k\times k}$, random vector $y_{k\times1}\sim(\mu,\Sigma)$, then $E(y^TAy)=tr(A\Sigma)+\mu^TA\mu$

(3) Proof (and proof the theorem):
$$
\begin{align}
\mathbb{E}(MS_{Res})&=\frac{1}{n-p}\mathbb{E}(e^Te)\\
&=\frac{1}{n-p}\mathbb{E}[y^T(I-H)^T(I-H)y]\\
&=\frac{1}{n-p}\mathbb{E}[y^T(I-H)y]\\
&=\frac{1}{n-p}\mathbb{E}[Tr(y^T(I-H)y)]\\
&=\frac{1}{n-p}\mathbb{E}[Tr(I-H)yy^T]\\
&=\frac{1}{n-p}\mathbb{E}(Tr[(I-H)(X\beta+\varepsilon)(X^T\beta^T+\epsilon^T)])\\
&=\frac{1}{n-p}\mathbb{E}(Tr[(I-H)(X\beta X^T\beta^T+\varepsilon\varepsilon^T)])\\
&=\frac{1}{n-p}\mathbb{E}(Tr[(I-H)\varepsilon\varepsilon^T)])\\
&=\frac{1}{n-p}Tr[(I-H)\mathbb{E}(\varepsilon\varepsilon^T)]\\
&=\frac{1}{n-p}Tr[(I-H)\Sigma]\\

\end{align}
$$
, since $y\sim (X\beta+0,\sigma^2I)$, so $\Sigma=\sigma^2I$

So 
$$
\begin{align}
\mathbb{E}(MS_{Res})&=\frac{1}{n-p}\sigma^2Tr(I-H)\\
&=\frac{1}{n-p}\sigma^2[Tr(I)-Tr(H)]\\
&=\frac{1}{n-p}\sigma^2(n-Tr[(X^TX)^{-1}X^TX])\\
&=\frac{1}{n-p}\sigma^2(n-p)\\
&=\sigma^2
\end{align}
$$
During the proof, we use assumptions (b) $\mathbb{E}(\varepsilon)=0$ (d) $Cov(e_i,e_j)\neq0$ for $i\neq j$ uncorrelated (e) $Var(\varepsilon_i)=\sigma^2$ constant for different i, or $Cov(\varepsilon)=\sigma^2I\Leftrightarrow(d)$ (f) No model mismatch

**2.4. Distribution of $\sigma^2$**

(1) Definition: Assume $\varepsilon\sim N(0,\sigma^2I)$, then $SS_{Res}/\sigma^2\sim \chi_{n-p}^2$

(2) We need **Theorem 2**: for constant $A_{k\times k}$ **idempotent** matrix and with rank $p'$, $y_{k\times 1}\sim N(\mu,\sigma^2I)$.Then $\frac{y^TAy}{\sigma^2}\sim\chi_{p',\lambda'}^{2'}$ where $\lambda'=\frac{\mu^T A\mu}{\sigma^2}$ as the noncertainty property

(3) Proof:

we know $y\sim N(X\beta_{true},\sigma^2I_n)$, 

so $p'=rank(I-H)=tr(I-H)$ by idempotent property, since eigenvalue is either 0 or 1

and $\lambda'=E[y]^T(I-H)E[y]/\sigma^2=0$ (展开H，得到分子为0), so $SS_{Res}/\sigma^2=y^T(I-H)y/\sigma^2\sim\chi_{n-p}^2$.

We can derive the proof by considering the [relationship](https://online.stat.psu.edu/stat414/lesson/16/16.5) between normal distribution and chi square distribution.

**2.5. Property of $\hat{\beta}$**

(1) $E(\hat{\beta})=\beta$

Proof: $E(\hat{\beta})=E[(X^TX)^{-1}X^Ty]=(X^TX)^{-1}X^TE(X\beta+\varepsilon)=\\(X^TX)^{-1}X^TX\beta=\beta$

(2) $Cov(\hat{\beta})=\sigma^2 C=\sigma^2(X^TX)^{-1}$

Proof: $Cov(\hat{\beta}_i,\hat{\beta}_j)=E[(\hat{\beta}_{LS}-E(\hat{\beta}_{LS}))(\hat{\beta}_{LS}-E(\hat{\beta}_{LS}))^T]\\=(X^TX)^{-1}X^TCov(y,y^T)X(X^TX)^{-1}\\=(X^TX)^{-1}X^TCov(\varepsilon,\varepsilon^T)X(X^TX)^{-1}=\sigma^2(X^TX)^{-1}$

, where we use assumptions  ==(b) $\mathbb{E}(\varepsilon)=0$ (d) $Cov(e_i,e_j)\neq0$ for $i\neq j$ uncorrelated (e) $Var(\varepsilon_i)=\sigma^2$ constant for different i, or $Cov(\varepsilon)=\sigma^2I\Leftrightarrow(d)$ (f) No model mismatch==

(3) By ==**Gauss-Markov Theorem**== above, $\hat{\beta}$ is the best linear unbiased estimator:

**Best**: smallest covariance matrix among any other $\hat{\beta}$, that is $D:=Cov(\hat{\beta})-Cov(\hat{\beta}_{LS})$ PSD

Proof: let $\hat{\beta}=[(X^TX)^{-1}X^T+D]y$ an unbiased estimator as $E[\hat{\beta}]=\beta_{true}$,

Then $E[\hat{\beta}]=E([(X^TX)^{-1}X^T+D]y)=\beta_{true}+XD\beta_{true}$, because $E[\hat{\beta}_{LS}]=\beta_{true}$, then we must have $DX=0$

Then $Cov(\hat{\beta})=\sigma^2[(X^TX)^{-1}X^T+D][(X^TX)^{-1}X^T+D]^T\\=\sigma^2[(X^TX)^{-1}+DD^T]$, as $DX=0$

So, $Cov(\hat{\beta})-Cov(\hat{\beta}_{LS})=\sigma^2DD^T$ PSD by definition where for any $z$ we have$Z^TDD^Tz\geq0$. This proof can be extended to $Cov(\varepsilon)=\Sigma$ in the general Gauss-Markov Theorem (See [Introduction to Linear Regression Analysis Fifth Edition Appendix C.11 (p. 597)](http://www.libgen.rs).

**Linear**: linear model in ${\beta}$

**Unbiased**: $E(\hat{\beta})=\beta$

----

#### 3. ML Estimation

**3.1. Optimization**

Assume $\varepsilon\stackrel{\text{i.i.d.}}{\sim} N(0,\sigma^2I_n)$, we have likelihood function
$$
p(y;\beta,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}exp[\frac{-(y-X\beta)^T(y-X\beta)}{2\sigma^2}]\sim N(X\beta_{true},\sigma^2I_n)
$$
Take log and first order, we have:

$\hat{\beta}_{ML}=(X^TX)^{-1}X^Ty$, $\hat{\sigma^2}_{ML}=\frac{(y-X\beta)^T(y-X\beta)}{n}=\frac{n-p}{n}\sigma^2_{LS}$, because we use one step to find them all, instead of p equations to find, so no need to lose p degree of freedom.

**3.2. Asymptotic property**

for $p(y;\theta)$, we have $\hat{\theta}\stackrel{a}{\sim}N(\theta,I^{-1}(\theta))$, under assumptions: (b) $\mathbb{E}(\varepsilon)=0$ (d) $Cov(e_i,e_j)\neq0$ for $i\neq j$ uncorrelated (e) $Var(\varepsilon_i)=\sigma^2$ constant for different i, or $Cov(\varepsilon)=\sigma^2I\Leftrightarrow(d)$ (f) No model mismatch. $I$ is fisher information, and $I^{-1}$ is Cramer-rao lower bound

----

#### 4. Hypothesis testing

**4.1. Assumptions**

$\varepsilon_i\stackrel{\text{i.i.d.}}{\sim} N(0,\sigma^2)$

**4.2. Test linear relationship between y and $x_{1,2,3,...}$, no need to find parameter**

**4.2.1. Hypothesis**

$H_0:\beta_1=...=\beta_k=0$, test with ANOVA
$$
\text{Corrected sum of squres: }SS_T=\sum_{i=1}^n(y_i-\bar{y})^2\\
\text{Regression/model sum of squares: }SS_R=\sum_{i=1}^n(\hat{y_i}-\bar{y})\\
\text{Residual sum of squares: }SS_{Res}=\sum_{i=1}^n(y_i-\hat{y}_i)^2
$$
$SS_T$ is total variance in the data, $SS_R$ is the variance in y that has been explained in the regression model, $SS_{Res}$ is the total variance hasn't been explained like noise or model mismatch. If the model is significant, then $SS_R>>SS_{Res}$

**4.2.2. Statistical testing**

We know, (1) $SS_{Res}/\sigma^2\sim\chi_{n-p}^2$ (**proof in section 2.4**), (2) $SS_R/\sigma^2\sim\chi_k^2$ if the null hypothesis is true (**proof in Appendix 3**), (3) $SS_R$ and $SS_{Res}$ are indepedent (**proof in Appendix 4**).

Under $H_0$: F test statistic is $F_0=\frac{SS_R/k}{SS_{Res}/{n-p}}\sim F_{k,n-p}$, if $F_0$ is large, we reject null hypothesis.

**4.2.3. Under $H_1$**

So denote $X_c=(I-1(1^T1)^{-1}1^T)X_R$, we can center the data. That is,
$$
X_R=\begin{bmatrix}
x_{11} & \cdots & x_{1k}\\
x_{21} & \cdots & x_{2k}\\
\vdots & \ddots & \vdots\\
x_{n1} & \cdots & x_{nk}\\
\end{bmatrix},X_c=\begin{bmatrix}
x_{11}-\bar{x}_1 & \cdots & x_{1k}-\bar{x}_k\\
x_{21}-\bar{x}_1 & \cdots & x_{2k}-\bar{x}_k\\
\vdots & \ddots & \vdots\\
x_{n1}-\bar{x}_1 & \cdots & x_{nk}-\bar{x}_k\\
\end{bmatrix}
$$
Then under $H_1$, $\lambda'=\frac{1}{\sigma^2}E^T(y)[H-1(1^t1)^{-1}1^T]E(y)=\frac{1}{\sigma^2}\beta_R^TX_c^TX_c\beta R^T\neq0$

Under $H_1$: $F_0\sim F_{k,n-p,\lambda'}$

**4.3. Test each parameter, need to find parameter**

**4.3.1. Hypothesis**

$H_0:\beta_j=0$, test with t test statistics 

**4.3.2. Statistical testing**.

If (1) $\frac{(n-p)MS_{Res}}{\sigma^2}=\frac{SS_{Res}}{\sigma^2}\sim \chi_{n-p}^2$ (**proof in 2.4**) (2) $MS_{Res}$ and $\hat{\beta_j}$ are independent (**proof in Appendix  5**, [ref1](https://math.stackexchange.com/questions/980363/if-y-x-beta-epsilon-prove-that-the-least-square-estimator-hat-beta-is-ind), [ref2](https://stats.stackexchange.com/questions/173396/estimators-independence-in-simple-linear-regression), [ref3](https://math.stackexchange.com/questions/2739109/why-are-y-and-haty-independent)). (3) $\hat{\beta}_j\sim N(\beta_j,\sigma^2(X^TX)^{-1}_{jj})$ (4) $MS_{Res}$ is unbiased estimator of $\sigma^2$

$t_0=\frac{\hat{\beta}_j}{\sqrt{MS_{Res}C_{jj}}}\sim t_{n-p}$, $C=(X^TX)^{-1}$

So, reject $H_0$ if $|t_0|>t_{c/2,n-p}$, may exists nonlinear relationship between $x_j$ and y

---

#### Appendix

**1. Show gradient in LS estimation**
$$
\begin{align}
\text{for } \frac{ S(\beta)}{ \beta}&=(y-X\beta)^T(y-X\beta)\\
&=f(\beta)^Tg(\beta)\\
\frac{\partial S(\beta)}{\partial \beta}&=(\nabla_{\beta}f(\beta)^T)g(\beta)+(\nabla_\beta g(\beta))^Tf(\beta)\\
&=\frac{\partial f(\beta)^T}{\partial \beta}g(\beta)+\frac{\partial g(\beta)^T}{\partial \beta}f(\beta)\\
&=-X^T(y-X\beta)-X^T(y-X\beta)\\
&=2[X^TX\beta-X^Ty]=0
\end{align}
$$
**2. Show the eigenvalue of idempotent matrix is either 0 or 1**

Proof: (1) For idempotent matrix $H$, we have eigenvalue $\lambda$ and eigenvector $v$, then by $Hv=v\lambda$

So, $HHv=Hv\lambda  \rightarrow Hv=(Hv)\lambda=v\lambda^2$

So, $v\lambda=v\lambda^2$, so $\lambda$ must be binary, either 0 or 1. Given $v$ is not 0 in real life.

(2) Moreover, we have $\sum_{i=1}^n\lambda_i=tr(H)=tr((X^X)^{-1}X^TX)=Tr(I_p)=p$, so there is $p$ numbers of $\lambda=1$ and $n-p$ numbers of $\lambda=0$. (Note the sum of eigen value equals to trace: proof on [Stackoverflow](https://math.stackexchange.com/questions/546155/proof-that-the-trace-of-a-matrix-is-the-sum-of-its-eigenvalues) and [Video](https://www.youtube.com/watch?v=OLl_reBXY-g), must-known knowledge from linear algebra)

**3. Prove $SS_R/\sigma^2\sim\chi_k^2$ under $H_0$**

We know, $\bar{y}=\frac{1}{n}1^Ty=(1^T1)^{-1}1^Ty$. Under $h_0$: $\beta_i=0$, for i=1,2,3,..., so $\beta=[\beta_0,0_{1\times k}]^T$, so $X\beta=\beta_01_{n\times 1}$
$$
\begin{align}
SS_R&=(\hat{y}-1\bar{y})^T(\hat{y}-1\bar{y})\\
&=y^T[H-1(1^T1)^{-1}1^T]^T[H-1(1^T1)^{-1}1^T]y\\
&=y^T[H-1(1^T1)^{-1}1^T]y
\end{align}
$$
, where $[H-1(1^T1)^{-1}1^T]$ is idempotent

By theorem 2, we have $p'=rank[H-1(1^T1)^{-1}1^T]=tr[H-1(1^T1)^{-1}1^T]=p-1=k$

$\lambda'=\frac{1}{\sigma^2}\beta_01^T[H-1(1^T1)^{-1}1^T]\beta_01\\=\frac{\beta_0^2}{\sigma^2}1^T[H1-1]=0$

Note: $H1=1$  is because $HX=X$ 

​	so $H[1,x_1,x_2,...,x_k]_{n\times p}=[1,x_1,x_2,...,x_k]$

​	So $[H1,Hx_1,Hx_2,...,Hx_k]=[1,x_1,x_2,...,x_k]$

​	So $H1=1$ for the frist term.

So, $SS_R/\sigma^2\sim \chi_k^2$

**4. Prove $SS_R$ and $SS_{Res}$ independent**

Need **Theorem 3**: constant $A_{k\times k}$ and $B_{k\times k}$, $y_{k\times 1}\sim N(\mu,\Sigma)$ (not iid), for $U=y^TAy$ and $V=y^TBy$. If $A\Sigma B=0_{k\times k}$, they are independent.

So, $SS_R=y^TAy$ and $SS_{Res}=y^TBy$, where $A=I-H$ and $B=H-1(1^T1)^{-1}1^T$ and if $\Sigma=\sigma^2I$, then $AB=0$ (refer to Appendix 3)

So, $SS_R$ and $SS_{Res}$ are independent.

**5. Prove $MS_{Res}$ and $\hat{\beta_j}$ are independent**

Need **Theorem 4**: constant $B_{q\times k}$, $ y_{k\times 1}\sim N(\mu,\Sigma)$, for $W=By$ and $U=y^TAy$. If $B\Sigma A=0_{q\times k}$, then $U$ and $W$ are independent.

So, $MS_{Res}=y^T(I-H)y/(n-p)$, $\hat{\beta}=(X^TX)^{-1}X^Ty$, then $\frac{\sigma^2}{n-p}(X^TX)^{-1}X^TI(I-H)=0$

So, $MS_{Res}$ and $\hat{\beta_j}$ are independent.