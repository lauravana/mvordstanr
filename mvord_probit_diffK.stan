data {
  int<lower=1> J;
  int<lower=1> Kmax;
  array[J] int<lower=1> K;
  array[J] int pos_alpha_beg;
  array[J] int pos_alpha_end;
  int<lower=1> P;
  int<lower=0> N;
  array[N, J] int<lower=1, upper=Kmax> y;
  array[N] vector[P] x;
}
parameters {
  matrix[J, P] beta;
  array[J] ordered[Kmax - 1] alpha_array;
  cholesky_factor_corr[J] L_Omega;
  array[N, J] real<lower=0, upper=1> u; // nuisance that absorbs inequality constraints
}
transformed parameters {
}
model {
  L_Omega ~ lkj_corr_cholesky(4);
  to_vector(beta) ~ normal(0, 5);
  // likelihood
  for (n in 1 : N) {
    vector[J] mu;
    vector[J] z;
    real prev;
    mu = beta * x[n];
    prev = 0;
    for (j in 1 : J) {
      vector[Kmax - 1] alphaj;
      alphaj = alpha_array[j];
      // Phi and inv_Phi may overflow and / or be numerically inaccurate
      if (y[n, j] == 1) {
        real ustar1;
        real t;
        ustar1 = Phi((alphaj[1] - (mu[j] + prev)) / L_Omega[j, j]);
        t = ustar1 * u[n, j];
        z[j] = inv_Phi(t); 
        target += log(ustar1); // Jacobian adjustment
      } else {
        if (y[n, j] == K[j]) {
          real ustarK; 
          real t;
          ustarK = Phi((alphaj[K[j] - 1] - (mu[j] + prev)) / L_Omega[j, j]);
          t = ustarK + (1 - ustarK) * u[n, j];
          z[j] = inv_Phi(t);
          target += log1m(ustarK); // Jacobian adjustment
        } else {
          real ustarku; 
          real ustarkl; 
          real t;
          ustarkl = Phi((alphaj[y[n, j] - 1]-(mu[j] + prev)) / L_Omega[j, j]);
          ustarku = Phi((alphaj[y[n, j]] - (mu[j] + prev)) / L_Omega[j, j]);
          t = ustarkl + (ustarku - ustarkl) * u[n, j];
          z[j] = inv_Phi(t); 
          target += log(ustarku - ustarkl); // Jacobian adjustment
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
generated quantities {
  corr_matrix[J] Omega;
  vector[sum(K) - J] alpha;
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  for (j in 1:J) {
    alpha[pos_alpha_beg[j]:pos_alpha_end[j]] = alpha_array[j][1 : (K[j] - 1)];
  }
}
