# 08 - Model Transform for Normal-Distribution Variance

### Empirical variance-stabilizing transform: residual has equal variance

To remedy correlation between $\varepsilon_i$ and $\varepsilon_j$ ($i\neq j$), by residual plot

<img src="/Users/apple/Library/Application Support/typora-user-images/Screen Shot 2021-03-25 at 11.16.59.png" alt="Screen Shot 2021-03-25 at 11.16.59" style="zoom:50%;" />

### Analytical variance-stabilizing transform: residual is Gaussian iid/equal variance

Box-Cox transformation procudure: 

1. Select $\lambda$, e.g. [0.1,0.2,0.3,...]

2. For iteration in $ \lambda$:

   1. Compute 
      $$
      y^{(\lambda)}=
      \begin{equation}\left\{ 
      \begin{aligned}
      \frac{y^\lambda-1}{\lambda[\prod_{i=1}^ny_i]^{(\lambda-1)/n}} ,\lambda\neq0\\
      [\prod_{i=1}^ny_i]^{1/n}\ln y,\lambda=0
      \end{aligned}
      \right.
      \end{equation}
      $$

   2. Fit $y^{(\lambda)}=X\beta+\varepsilon'$
   3. Find $SS_{Res}(\lambda)$

3. Plot $SS_{Res}(\lambda)\sim\lambda$ and find $\lambda^*$ to minimize the $SS_{Res}$

4. 在$\lambda^*$ 领域缩小范围找, repeat 1

### Transformations

(1) log (2) exponential (3) reciprocal (e.g. $y=x/(\beta_0x-\beta_1)$)

