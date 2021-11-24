//
// Zero (non-response, really) inflated ordinal regression
//
// This model is a mixture of ordered logistic regression and regular logistic regression.
// Ordered categorical regression is very useful for survey work, since respondents are
// often asked to choose one of several ordered responses (e.g. a Likert scale ranging from
// Strongly disagree to Strongly agree). But surveys also have non-response. For non-ordered
// responses, this can be managed through a multinomial likelihood. But that's not ideal
// for ordered responses.
//
// Here, we use an ordered logistic likelihood to model the ordered part of the responses,
// and logistic regression to model the probability of non-response (coded as 0's). You can 
//choose to include predictors for both.
//
// Last updated: Nov 7, 2021
// Author: Octavio Medina
//

data {
  int<lower=2> K; // number of categories
  int<lower=0> N; // number of observations
  int<lower=1> D; // number of predictors
  int<lower=0,upper=K> y[N]; // response variable
  matrix[N, D] x; // design matrix
}
parameters {
  vector[D] beta; // predictors for ordered logistic part (e.g. likert scale)
  ordered[K-1] c; // number of thresholds for ordered logistic
  real alpha_p; // intercept for proportion of non-response
  vector[D] beta_p; // predictors for proportion of non-response
}
model {
  // model
  for (n in 1:N) {
    if (y[n] == 0) // assume non-response coded as 0
      target += bernoulli_logit_lpmf(1|alpha_p + x[n] * beta_p); // predict non-response with probability p
    else // otherwise use ordered logistic with prob 1-p
      target += bernoulli_logit_lpmf(0|alpha_p + x[n] * beta_p) + ordered_logistic_lpmf(y[n]| x[n] * beta, c);
  }
}
generated quantities {
  vector[N] yrep; // generate samples
  for (n in 1:N) {
    if (bernoulli_rng(inv_logit(alpha_p + x[n] * beta_p)) == 1)
      yrep[n] = 0;
    else
      yrep[n] = ordered_logistic_rng(x[n] * beta, c);
  }
}
