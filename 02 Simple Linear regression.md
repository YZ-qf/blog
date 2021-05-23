# Simple Linear regression

### 1. Function

$y=f(x;\beta_0;\beta_1)=\beta_0+\beta_1x+\varepsilon$, where (1) $\varepsilon\sim iid(0,\sigma^2)$, and $\sigma^2$ is unkown, even if $\mathbb{E}(\varepsilon)\neq 0$ we can demean it and put the part to $\beta_0$ (2) x is input/regressor, (3) y is output/rersponse. 

So, x is deterministic, but y and $\varepsilon$ are random. Our goal is to approximate y with $\tilde{y}$ by fitting $f(x;\theta)$

### 2. Meaning

the linear regression model is called linear because $f(x;\beta)$ is linear of the parameters $\{\beta_i\}$ 

### 3. Parameter estimation

**3.1. General procedure:**

(1) select a measure of fitness (2) optimize the measure with respect to the model parameters for a data set

**3.2. Classic methods: **

Least Square (LS), Maximum likelihood (ML).

**3.3. Note:**

Parameter estimate is the value from estimation on a specific data; Parameter estimator is Random Variable (RV); So, estimate is the realization of estimator

#### 3.4. LS estimation:

$$
[\beta_0,\beta_1]=\arg\min_{\beta_0,\beta_1}S(\beta_0,\beta_1)=\min_{\beta_0,\beta_1}\sum_{i=1}^n(y_i-\beta_0-\beta_1x_i)^2
$$

(1) Therefore, we have $\hat{\beta_0}=\bar{y}-\hat{\beta_1}\bar{x},\hat{\beta_1}=\frac{Cov(x,y)}{Var(x)}=\frac{S_{xy}}{S_{xx}}$

(2) Residual is $e_i=y_i-\hat{y}_i=y_i-(\hat{\beta_0}+\hat{\beta_1}x_i)$

(3) Residual/error sum of squares is $SS_{Res}=\sum_{i=1}^ne_i^2$

(4) Estimator of $\sigma^2$ is $\hat{\sigma}^2=\frac{SS_{Res}}{n-p}=MS_{Res}$, here $p$ is 2 because there are only two parameters $\beta_0,\beta_1$, the proof is in appendix

(5) if $\varepsilon\sim N(0,\sigma^2)$, then $\hat{\beta_1}\sim N(\beta_{10},\frac{\sigma^2}{S_{xx}})$, $\hat{\beta}_0\sim N(\beta_{00}, \sigma^2(\frac{1}{n}+\frac{\bar{x}^2}{S_{xx}}))$, proof is shown in appendix.

(6) 6 Assumptions for OLS: (a) regression model is linear in coefficients and error term (b) $\mathbb{E}(\varepsilon)=0$ (c) All independent variables are uncorrelated with the error term (d) $Cov(e_i,e_j)\neq0$ for $i\neq j$ (e) $Var(\varepsilon_i)=\sigma^2$ constant for different i (f) No model mismatch

#### 3.5. ML Estimation

LS estimation does not need to assume underlying distribution, but ML needs to assume. So, the goal is to best approximate the distribution. Usually, there is no close form expression for $\hat{\theta}_{ML}$ except for Gaussian error distribution.
$$
\hat{\theta}_{ML}=\max_{\theta}p(y;\theta)
$$
3.5.1. Gaussian error with two parameters:

We have $l(\beta)=\ln p(y;\Beta,\sigma^2)=\ln\prod_{i=1}^n\frac{1}{\sqrt{2\pi\sigma^2}}exp[\frac{-(y_i-\beta_0-\beta_1x_i)^2}{2\sigma^2}]$.

So take gradient of $l$ we can find $\hat{\beta}_0=\bar{y}=\hat{\beta}_1\bar{x}$, $\hat{\beta}_1=\frac{S_{xy}}{S_{xx}}$, $\hat{\sigma^2}=\frac{\sum_{i=1}^n(y_i-\hat{\beta_0}-\hat{\beta_1})^2}{n}$ which is a biasaed estimator of $\sigma^2$. 

3.5.2. Note: such regression is parametric model. and these two are deterministic linear parametric regression model (just $\beta,\sigma$ unknown but follows certain dsitribution)

**3.6. Hypothesis testing for 2 parameter**

(1) To test $H_0:\hat{\beta}_1=\beta_{10}$ for $\varepsilon\sim \text{iid } N(0,\sigma^2)$, we need test statistics $t_0=\frac{\hat{\beta}_1-\beta_{10}}{\sqrt{MS_{Res}/S_{xx}}}\sim t_{n-2}$. Proof is in Appendix (8)

(2) To test $H_0:\hat{\beta}_0=\beta_{00}$ for $\varepsilon\sim \text{iid } N(0,\sigma^2)$, we need test statistics $t_0=\frac{\hat{\beta}_0-\beta_{00}}{\sqrt{MS_{Res}(1/n+\bar{x}^2/S_{xx})}}\sim t_{n-2}$, so we reject $ H_0$ if $|t_0|>t_{c/2,n-2}$



----

==Appendix==

Prove (given in the manuscript):

1  The sum of the residuals in the assumed model, cf. Eq.(1), is always zero.

2  The sum of the observed values yi equals the sum of the fitted values.

3  The LS regression line always passes through the centroid of the data.

4  The sum of the residuals weighted by the corresponding value of the regressor variable always equal to zero.

5 The sum of the residuals weighted by the corresponding fitted value always equals zero, that is $\sum_{i=1}^n\hat{y}_ie_i=0$

6 prove (4)  $MS_{Res}$ is unbiased estimator of $\sigma^2$, 直接展开 $y_i$ and $\hat{y}_i$

7 prove (5) the formula

8 prove 3.6. (a) $\frac{(n-2)MS_{Res}}{\sigma^2}\sim \chi_{n-2}^2$, and (b) $MS_{Res}$ and $\hat{\beta}_1$ are independent

