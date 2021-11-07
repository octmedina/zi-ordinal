
library(readr)
library(dplyr)
library(rstan)

data <- read_csv("/Users/octaviomedina/Desktop/Desktop Archive/Thoughts/Github/zi-ordinal/merkel_data.csv")

D <- 4 # number of covariates
N <- length(data$confid_merkel) # number of observations
y <- data$confid_merkel # response variable
x <- matrix(c(data$party, # covariates
              data$edu,
              data$race,
              data$income), 
            nrow = N, ncol = D) 
K <- length(unique(data$confid_merkel)) - 1 # number of ordinal levels (0 doesn't count)


stan_data <- list(
  N = N,
  K = K,
  y = y, 
  x = x, 
  D = D)


fit_merkel <- stan(file = "https://raw.githubusercontent.com/octmedina/zi-ordinal/main/zi-ordinal-model.stan", data = stan_data,
                   chains = 2,
                   cores = 2,
                   iter = 2000)


print(fit_merkel, pars= c("beta", "c", "alpha_p", "beta_p"))

