# 07 - Residual Plots

### Normal probability plot

PDF + CDF vs t-distribution plot. n>30 gives stable **normal distribution** (otherwise there is no linearity around 0 on the residual plot)

**Procedure**: (1) Sort R-student Residuals (2) Plot against cummulative probability $i/n$ or $ (i-0.5)/n$, for i=1,2,...n

**Result**: the points should lie on a straight line between 0.33 to 0.67.

Note, (i) Small departures from the normality assumption do not affect the model greatly, but gross nonnormality is potentially more serious as the t or F statistics and confidence and prediction intervals depend on the normality assumption. (ii) Furthermore, if the errors come from a distribution with thicker or heavier tails than the normal, the least-squares fit may be sensitive to a small subset of the data. (iii) Heavy-tailed error distributions often generate outlier that “pull” the least-squares fit too much in their direction. (iv) **But** , fitting the parameters tends to destroy the evidence of nonnormality in the residuals, and consequently we cannot always rely on the normal probability plot to detect departures from normality.

### Plot of R-St. residuals vs fitted values

test the relationship between $e_i$ and $\hat{y}_i$. If statistics of residual as $t_i\sim \hat{y}_i$ plot that $t_i$ does not sit in a horinzontal band (uniformly distributed, not condense), then (1) $var(\varepsilon_1)\neq var(\varepsilon_2)\neq ...$  not independent, not iid, **variance not constant**. (2) model mismatch

Solution, (a) use transformation on input/output (b) use weighted least square.

(3) the nonlinearity between output and input, e.g. forget to include the square term in the model.

Note,we plot $e_i\sim \hat{y}_i$ not $t_i\sim y_i$, proof in **Appendix 1**.



### Plot of residuals vs regressor

Useful

### Test lack of fit

Whether including new parameters can reduce $e_i$ or not? To do it, we need replicate observations on the response y for certain x.

SS(Res) = true error + model mismatch = SS(Pure Error) + SS(Lack of Fit). That is, $y_{ij}-\hat{y}_i=(y_{ij}-\bar{y}_i)+(\bar{y}_i-\hat{y}_i)$. Then take square on both side.

Then, we do F test on SS_Res/SS_PureError. $\frac{SS_{LOF}}{\sigma^2}\sim \chi_{m-p}^2,\frac{SS_{PE}}{\sigma^2}\sim \chi_{n-m}^2,F_0=\frac{SS_{LOF}/(m-p)}{SS_{PE}/(n-m)}\sim F_{m-p,n-m}$,

## Appendix

#### Appendix 1

Denote 
$$
v=\begin{bmatrix}e\\\hat{y}\end{bmatrix}=\begin{bmatrix}I-H\\H\end{bmatrix}y\\
So,cov(v)=\sigma^2\begin{bmatrix}I-H&0\\0&H\end{bmatrix}
$$
So, $cov(e,\hat{y})=0$

Or, directly
$$
\begin{align}
cov(e,\hat{y})&=cov(y-\hat{y},\hat{y})\\
&=E[(y-X(X^TX)^{-1}X^Ty)^T(X(X^TX)^{-1}X^Ty)]\\
&=E[y^TX(X^TX)^{-1}X^Ty-y^TX(X^TX)^{-1}X^Ty-y^TX\beta+y^TX\beta]\\
&=0
\end{align}
$$
