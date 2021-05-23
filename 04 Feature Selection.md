# 04 - Feature Selection

### Standardized method for input and output

(1) Unit length scaling: $\frac{x_{ij}-\bar{x}_j}{\sqrt{\sum_{i=1}^n(x_{ij-\bar{x_j}})^2}}$ so that sum of squares of the scaled variable = 1

(2) Unit normal scaling: $\frac{x_{ij}-\bar{x}_j}{\sqrt{\frac{\sum_{i=1}^n(x_{ij}-\bar{x}_j)^2}{n-1}}}$

(3) Winsorizing large values: cut off outliers

(4) Log transform for large values

### Missing value

(1) Discard

(2) Replace with sample mean

(3) Imputation

### Generate new features

(1) Product, interaction

(2) Random feature

### Feature selection and regularization

(1) Pearson correlation coefficient (PCC) $P(i)=\frac{Cov(X_i,Y)}{\sqrt{var(X_i)var(Y)}}$: detect linear relationship

(2) maximum correlation by $\max_{f,g}\mathbb{E}[f(X_i)g(Y)]$ to disclose nonlinear relationship