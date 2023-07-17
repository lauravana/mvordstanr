data {
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> P;
  int<lower=0> N;
  array[N, J] int<lower=1, upper=K> y;
  array[N] vector[P] x;
}
parameters {
  matrix[J, P] beta;
  ordered[K-1] alpha;
  cholesky_factor_corr[J] L_Omega;
  array[N, J] real<lower=0, upper=1> u; // nuisance that absorbs inequality constraints
}
model {
  L_Omega ~ lkj_corr_cholesky(4);
  to_vector(beta) ~ normal(0, 5);
  // implicit: u is iid standard uniform a priori
  {
    // likelihood
    for (n in 1 : N) {
      vector[J] mu;
      vector[J] z;
      real prev;
      mu = beta * x[n];
      prev = 0;
      for (j in 1 : J) {
        // Phi and inv_Phi may overflow and / or be numerically inaccurate
        if (y[n, j] == 1) {
          real boundu; // threshold at which utility = 0
          real t;
          boundu = Phi((alpha[1]-(mu[j] + prev)) / L_Omega[j, j]);
          t = boundu * u[n, j];
          z[j] = inv_Phi(t); // implies utility is negative
          target += log(boundu); // Jacobian adjustment
        } else {
          if (y[n, j] == K) {
            real boundl; // threshold at which utility = 0
            real t;
            boundl = Phi((alpha[K-1]-(mu[j] + prev)) / L_Omega[j, j]);
            t = boundl + (1 - boundl) * u[n, j];
            z[j] = inv_Phi(t); // implies utility is positive
            target += log1m(boundl); // Jacobian adjustment
          } else {
            real boundll; // threshold at which utility = 0
            real bounduu; // threshold at which utility = 0
            real t;
            boundll = Phi((alpha[y[n, j] - 1]-(mu[j] + prev)) / L_Omega[j, j]);
            bounduu = Phi((alpha[y[n, j]]-(mu[j] + prev)) / L_Omega[j, j]);
            t = boundll + (bounduu - boundll) * u[n, j];
            z[j] = inv_Phi(t); // implies utility is positive
            target += log(bounduu - boundll); // Jacobian adjustment
          }
          
        }
        if (j < J) {
          prev = L_Omega[j + 1, 1 : j] * head(z, j);
        }
        // Jacobian adjustments imply z is truncated standard normal
        // thus utility --- mu + L_Omega * z --- is truncated multivariate normal
      }
    }
  }
}
generated quantities {
  corr_matrix[J] Omega;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
}
