# 09 - Generalized Lienar Least Square

#### Purpose:

guess with boxcox and residual plot is too empirical. But if we know $V$, we know full property of $\varepsilon$, then no model mismatch

#### Function: 

$y=X\beta+\varepsilon$, where $E(\varepsilon)=0,Cov(\varepsilon)=\sigma^2V$, $\sigma^2$ unknown and $V$ known (we assume one by ourselves). We can generalize $Cov(\varepsilon)=\Sigma$ and still can have same unbiased estimators, proof is omitted in this note (see textbook C11).

Note, $V$ is positive definite (PD)

#### Optimization

Denote $V=K^TK=KK$ where $K$ is PD. Denote $z=K^{-1}y,B=K^{-1}X,g=K^{-1}\varepsilon$

Regress $z=B\beta+g$ where $E(g)=0,Var(g)=\sigma^2I$, proof in Appendix 1. Note, now we can use Gauss Markov Thereom and do least square estimation. Use least square estimation.

We have $\hat{\beta}=(X^TV^{-1}X)^{-1}X^TV^{-1}y$ and $E(\hat{\beta})=\beta$ and $Cov(\hat{\beta})=\sigma^2(X^TV^{-1}X)^{-1}$

Q: by least square estimation, GLS is min-covariance, LS is also min-covariance, is it true? No, here, LS is not BLUE.

Note, we can add constraint to least square optimization to make the fitting more robust.

#### Residual

$SS_{Res}=y^TAy$, $A=V^{-1}-V^{-1}X(X^TV^{-1}X)^{-1}X^TV^{-1}$. By theorem 1 in **03 Multiple Linear Regression**, $E(SS_{Res})=tr(\sigma^2VA)+(X\beta)^TAX\beta=\sigma^2(n-p)$, $E(MS_{Res})=\sigma^2$

## Appendix

#### Appendix 1

$$
\begin{align}
Var(g)&=E[[K^{-1}\varepsilon-E(g)][K^{-1}\varepsilon-E(g)]^T]\\
&=E[K^{-1}\varepsilon\varepsilon^TK^{-1}]\\
&=\sigma^2K^{-1}VK\\
&=\sigma^2I
\end{align}
$$

