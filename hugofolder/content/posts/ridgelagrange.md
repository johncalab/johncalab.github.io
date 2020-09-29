+++
title = "Why lasso kills features while ridge regression makes them small?"
tags = ["machine learning", "math", "regression", "interviews"]
date = "2020-09-28"
draft = true
+++
While I was prepping for Data Science interviews during my fellowship at [Insight](https://insightfellows.com/), a question that came up often was: "why does lasso tends to make kill features, while ridge makes them small?"
Or, put differently, "why is lasso the right choice for feature selection?"
The answer to this question is actually a bit subtle.

Here is the plan for this post:
* Set up the problem and notation, recall what ridge and lasso are.
* Provide handwavy intuition.
* Discuss why I don't think saying "Lagrange multipliers" actually answers the question.
* Provide a Bayesian point of view, which actually provides an explanation (in the Gaussian case).

# Ridge and Lasso
In a supervised ML problem we have _inputs_ {{< katex >}}x_i{{< /katex >}} and _outputs_ {{< katex >}}y_i{{< /katex >}}. 
To describe the relationship between the two, we then choose a model

{{< katex display>}}\hat y_i := f(x_i;w){{< /katex >}}

Here {{< katex >}}\hat y_i{{< /katex >}} are the _predictions_, which hopefully are close to the true values {{< katex >}}y_i{{< /katex >}}.
The function {{< katex >}}f{{< /katex >}} is the chosen modeling function, which depends on the input {{< katex >}}x{{< /katex >}} and the _weights_ {{< katex >}}w \in \R^k{{< /katex >}}.
Depending on your background, the weights {{< katex >}}w{{< /katex >}} are called _parameters_ and denoted by {{< katex >}}\theta{{< /katex >}}.
For example, in linear regression we would have

{{< katex display>}}f(x;w) = w^\top x{{< /katex >}}

Finally, we also choose a _loss function_ {{< katex >}}l(\hat y_i,y_i){{< /katex >}} to measure how good our model is.

Once all these choices have been made, it's time to _train_ our model, i.e. find the best weights:

{{< katex display>}}
    w^*
        = \argmin_{w \in \R^d} J(w)
        = \argmin_{w \in \R^d} \frac 1 n \sum_{i=1}^n l(\hat y_i, y_i) 
        = \argmin_{w \in \R^d} \frac 1 n \sum_{i=1}^n l(f(x_i;w), y_i) 
{{< /katex >}}

where {{< katex >}}n{{< /katex >}} is the number of samples {{< katex >}}x_1,\ldots,x_n{{< /katex >}} and {{< katex >}}y_1,\ldots,y_n{{< /katex >}}.
We will call this "total loss function" the _objective_ (or _cost_) function:

{{< katex display >}}
J(w) := \frac 1 n \sum_{i=1}^n l(\hat y_i, y_i)
{{< /katex >}}

### Overfitting
What if the model is overfitting?
This is where ridge and lasso come into play.
They both are simple variants of the optimization problem above.
In ridge regression, aka {{< katex >}}L_2{{< /katex >}} regression, the objective function is

{{< katex display >}}
    J_\text{ridge}(w) 
        = J(w) + \lambda \left( \sum_{j=1}^k w_j^2 \right)
        = J(w) + \lambda  \Vert w \Vert_2^2
{{< /katex >}}

while in lasso regression, aka {{< katex >}}L_1{{< /katex >}} regression, we have

{{< katex display>}}
    J_\text{lasso}(w) 
        = J(w) + \lambda \left( \sum_{j=1}^k \vert w_j \vert \right)
        = J(w) + \lambda \Vert w \Vert_1
{{< /katex >}}

where {{< katex >}}\lambda \geq 0{{< /katex >}} is the _regularization parameter_, and the _{{< katex >}}p{{< /katex >}}-norm_ of a vector {{< katex >}}w \in \R^k{{< /katex >}} is

{{< katex display>}}
    \Vert w \Vert_p := \left( \sum_{i=1}^k \vert w_k \vert^p \right)^{\frac 1 p}
{{< /katex >}}

The parameter {{< katex >}}\lambda{{< /katex >}} is a _hyperparameter_, ie it is not part of the optimization problem, and must be chosen at the start (just as we chose the function {{< katex >}}f{{< /katex >}}, and the data {{< katex >}}x_i,y_i{{< /katex >}} to train the model).

# Intuition
Intuitively, the extra factor {{< katex >}}\lambda \Vert w \Vert{{< /katex >}} in our objective function is a _penalty_: it punishes {{< katex >}}w{{< /katex >}} for being big.
A bigger coefficient {{< katex >}}\lambda{{< /katex >}} leads to a harsher penalty: when {{< katex >}}\lambda = 0{{< /katex >}} we recover the original optimization problem, when {{< katex >}}\lambda \to \infty{{< /katex >}} the only solution will be {{< katex >}}w = 0{{< /katex >}}.

So what is the difference between ridge and lasso?
For simplicity, imagine we are dealing with a single weight {{< katex >}}w \in \R{{< /katex >}}.
If {{< katex >}}w = 10^{-m}{{< /katex >}}, then {{< katex >}}J_\text{lasso}(w) = J(w) + \lambda 10^{-m}{{< /katex >}}.
However, {{< katex >}}J_\text{ridge}(w) = J(w) + \lambda 10^{-2m}{{< /katex >}}.
Hence if {{< katex >}}w{{< /katex >}} is already small, ridge will stop penalizing it (for example if {{< katex >}}10^{-2m}{{< /katex >}} is already below the numerical precision of the software being used).
Meanwhile with lasso {{< katex >}}w{{< /katex >}} will still be penalized.

# Lagrange Multipliers

I don't blame you if you're not convinced by the handwavy intuitive explanation given above.
How can we be more rigorous?
Another justification I've heard comes from Lagrange multipliers.
If we look at the equation for {{< katex >}}J_\text{ridge}{{< /katex >}} again, it does look like the Lagrangian of a constrained optimization problem.
If you need to brush up on Lagrange multipliers, I recommend having a look at the [wikipedia page](https://en.wikipedia.org/wiki/Lagrange_multiplier).

Now, {{< katex >}}J_\text{ridge}(w){{< /katex >}} is the Lagrangian of a constrained optimization problem.
Where the function to minimize is {{< katex >}}J(w){{< /katex >}}, subject to the constraint {{< katex >}}\Vert w \Vert_2^2 = c{{< /katex >}}, for some fixed value {{< katex >}}c \in \R{{< /katex >}}.
Same story for lasso.
However, finding the minima of {{< katex >}}J_\text{ridge}(w){{< /katex >}} with respect to {{< katex >}}w{{< /katex >}} is _not_ the same as solving the constraint optimization problem, for two reasons.

1. The hyperparameter {{< katex >}}\lambda{{< /katex >}} is fixed, while in Lagrange multipliers it's a free variable.
1. In Lagrange multipliers you look for _critical points_, not minima. 
Indeed, the critical points corresponding to minima of the constrained problem are actually always _saddle_ points of the Lagrangian.

But let's pretend these technical difficulties weren't there, and we were actually dealing with a constrained optimization problem.
What would we have gained?
In ridge, we would be saying the solution should live on a sphere {{< katex >}}S_2 = \Vert w \Vert_2^2 = c{{< /katex >}}, while in lasso it should live on a tilted cube {{< katex >}}S_1 = \Vert w \Vert_1 = c{{< /katex >}}.
How does this imply lasso is better for feature selection?
It really doesn't.

For example in two dimensions {{< katex >}}S_2{{< /katex >}} contains exactly four points with at least one component being zero.
And the same holds for {{< katex >}}S_1{{< /katex >}}.

# Bayesian point of view

Let's change perspective and put an end to this ridge vs lasso dilemma!
Instead of inputs {{< katex >}}x_i{{< /katex >}} and outputs {{< katex >}}y_i{{< /katex >}}, let's say we are dealing with random variables {{< katex >}}x,y{{< /katex >}} (I'm using the terms random variable and random vector interchangeably).

We assume a relationship between the variable {{< katex >}}y{{< /katex >}}, the variable {{< katex >}}x{{< /katex >}}, and some weights {{< katex >}}w{{< /katex >}}, which we model as a random variable as well (as in this section we are being Bayesian).
Instead of picking a model {{< katex >}}f(x,w){{< /katex >}}, we want to directly study the probability distribution of the _posterior_ {{< katex >}}P(w|x,y){{< /katex >}}.
Finally, by Bayes, we have {{< katex >}}P(w|x,y) = \frac {P(y|w,x)P(w)} {P(x,y)}{{< /katex >}}.

The MAP principle (Maximum a Posteriori, which is the Bayesian version of MLE, maximum likelihood estimation) states that we should estimate the weights {{< katex >}}w{{< /katex >}} as

{{< katex display>}}
    w^* 
        = \argmax_w P(w|x,y)
        = \argmax_w \frac {P(y|w,x)P(w)} {P(x,y)}
        = \argmax_w P(y|w,x)P(w)
{{< /katex >}}

Here comes the crucial assumption: the distribution {{< katex >}}P(y|x,w) \sim N(\mu, \Sigma){{< /katex >}} is assumed to be Gaussian.
This might seem like an incredibly strong assumption, but this is precisely what happens in linear regression.
Now, we don't really know what {{< katex >}}\mu{{< /katex >}} and {{< katex >}}\Sigma{{< /katex >}} are.
And this is where the collected data {{< katex >}}x_i,y_i{{< /katex >}} come in.

We assume the observed data {{< katex >}}x_i,y_i{{< /katex >}} are iid samples of the variables {{< katex >}}x,y{{< /katex >}}.



Here comes the crucial assumption.



{{< katex >}}{{< /katex >}}