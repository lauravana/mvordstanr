library(MASS)

K <- 3;
J <- 3;
N <- 1500;
P <- 2

# stick breaking construction to generate a random correlation matrix
L_Omega <- matrix(0, J, J);
L_Omega[1,1] <- 1;
for (i in 2:J) {
  bound <- 1;
  for (j in 1:(i-1)) {
    L_Omega[i,j] <- runif(1, -sqrt(bound), sqrt(bound));
    bound <- bound - L_Omega[i,j]^2;
  }
  L_Omega[i,i] <- sqrt(bound);
}


Omega <- matrix(c(1, 0.9, 0.8, 0.9, 1, 0.7, 0.8,0.7,1), 3)

#Omega <- L_Omega %*% t(L_Omega);

x <- matrix(rnorm(N * P, 0, 1), N, P);

beta <- matrix(rep(c(1,-1), J), J, P, byrow = TRUE);

z <- matrix(NA, N, J);
for (n in 1:N) {
  z[n,] <- mvrnorm(1, x[n,] %*% t(beta), Omega);
}

alpha <- c(-1, 1)
y <- matrix(0, N, J);
for (n in 1:N) {
  for (j in 1:J) {
    y[n,j] <- cut(z[n,j], c(-Inf, alpha, Inf))
  }
}

library(cmdstanr);

DATA <- list(K = max(y), P = P, J = J, N = N, y = y, x = x)

mod <- cmdstan_model("mvord_probit.stan")

fit <- mod$sample(
  data = DATA,chains = 1,
  step_size=0.01, adapt_delta=0.99,
  init = list(list(beta = matrix(0, nrow = J, ncol = P))));

print(fit$summary(variables = c("beta", 
                                "alpha",
                                "Omega",
                                "lp__"), "mean", "sd"), n = 30)

as.mcmc.list <- function(fit) {
  sample_matrix <- fit$draws()
  class(sample_matrix) <- 'array'
  n_chain <- dim(sample_matrix)[[2]]
  mcmc_list <- coda::as.mcmc.list(
    lapply(seq_len(n_chain),
           function(chain) coda::as.mcmc(sample_matrix[, chain, ])))
  return(mcmc_list)
}
pdf(file='test-plot.pdf')
plot(as.mcmc.list(fit))
dev.off() 

###################################################
## Different number of classes for each response ##
###################################################

alpha <- list(c(-1, 1), c(-1, 0, 1), 0)
y <- matrix(0, N, J);
for (n in 1:N) {
  for (j in 1:J) {
    y[n,j] <- cut(z[n,j], c(-Inf, alpha[[j]], Inf))
  }
}

DATA <- list(Kmax = max(y), 
             K = apply(y, 2, max),
             Ksum = sum(apply(y, 2, max)),
             P = P, J = J, N = N, y = y, x = x)

mod <- cmdstan_model("mvord_probit_diffK.stan")

fit <- mod$sample(
  data = DATA,chains = 1,
  step_size=0.01, adapt_delta=0.99,
  init = list(list(beta = matrix(0, nrow = J, ncol = P))));

print(fit$summary(variables = c("beta", 
                                "alpha",
                                "Omega",
                                "lp__"), "mean", "sd"), n = 30)



