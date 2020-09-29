+++
title = "How do lasso and ridge help constraining feature size?"
tags = ["machine learning", "math", "interviews", "optimization"]
date = "2020-09-28"
draft = false
+++
Here is a simple fact that escaped me when learning basic Machine Learning stuff.
To help with overfitting when training a supervised ML model, a standard technique is to _regularize_ by adding a penalty term, for example in ridge and lasso regression.
Let's recall how this works.

We start out by wanting to find the following minimum:

{{< katex display>}}
    (+) \quad 
    w^* 
        := \argmin_w \frac 1 n \sum_{i=1}^n l(\hat y_i, y_i)
        = \argmin_w \sum_{i=1}^n l(f(x_i;w), y_i)
        = \argmin_w J(w)
{{< /katex >}}

where

* {{< katex >}}x_i,y_i{{< /katex >}} are the gathered datapoints, 
* {{< katex >}}f(x;w){{< /katex >}} is the chosen ML model, 
* {{< katex >}}w \in \R^k{{< /katex >}} are the _weights_ (or _parameters_) of the model, 
* {{< katex >}}\hat y_i = f(x_i;w){{< /katex >}} is the model prediction, 
* {{< katex >}}l{{< /katex >}} is the chosen _loss function_ (e.g. sum of squares),
* {{< katex >}}J(w) = \sum_{i=1}^n l(f(x_i;w), y_i){{< /katex >}} is just convenient notation, which we call the _cost_ (or _objective_) function.

To help with overfitting, a basic strategy one learns is to modify equation {{< katex >}}(+){{< /katex >}} by adding a _penalty_ term.
For example in _ridge regression_ we would have

{{< katex display>}}
     \quad w_\text{ridge}^* 
        := \argmin_w J_\text{ridge}(w)
        := \argmin_w J(w) + \lambda \sum_j w_j^2
{{< /katex >}}

where {{< katex >}}w_j{{< /katex >}} is the {{< katex >}}j\text{-th}{{< /katex >}} component of the vector {{< katex >}}w{{< /katex >}}, and {{< katex >}}\lambda > 0{{< /katex >}} is a fixed hyperparameter (i.e. a parameter which is chosen at the start, not learned from the data).

More generally, we could consider the optimization problem

{{< katex display>}}
    (1) \quad
        \argmin_w J(w) + \lambda \Vert w \Vert_p^p
{{< /katex >}}

where recall the _{{< katex >}}p\text{-norm}{{< /katex >}}_ of the vector {{< katex >}}w{{< /katex >}} is {{< katex >}}\Vert w \Vert_p := \left( \sum_{j=1}^k \vert w \vert^p \right)^{\frac 1 p} {{< /katex >}}.
For {{< katex >}}p=2{{< /katex >}} we recover ridge, and for {{< katex >}}p=1{{< /katex >}} we have what is called the _lasso_.

It turns out that the optimization problem is equivalent to the problem

{{< katex display>}}
    (2) \quad
        \argmin_{w, \Vert w \Vert_p^p \leq r} J(w)
{{< /katex >}}

for some fixed {{< katex >}}r{{< /katex >}}.
So a given choice of {{< katex >}}\lambda{{< /katex >}} will determine the exact value of {{< katex >}}r{{< /katex >}}, and viceversa.

# Proof

Let's show how the problems {{< katex >}}(1), (2){{< /katex >}} are indeed equivalent.
Caveat: I will also assume the function {{< katex >}}J(w){{< /katex >}} to be bounded below.
This is often satisfied, as the loss function {{< katex >}}l{{< /katex >}} is typically non-negative.
Without loss of generality we may then assume that {{< katex >}}J(w) > 0{{< /katex >}} for all {{< katex >}}w{{< /katex >}}.
Finally, let's write {{< katex >}}g(w) := \Vert w \Vert_p^p{{< /katex >}} to ease notation.

### {{< katex >}}(1){{< /katex >}} implies {{< katex >}}(2){{< /katex >}}

Suppose {{< katex >}}w^* {{< /katex >}} is a solution to {{< katex >}}(1){{< /katex >}} for a fixed {{< katex >}}\lambda{{< /katex >}}.
We claim {{< katex >}}w^* {{< /katex >}} is a solution to {{< katex >}}(2){{< /katex >}} for a fixed {{< katex >}}r{{< /katex >}}.
The first guess for what {{< katex >}}r{{< /katex >}} should be is the correct one.
Indeed, let {{< katex >}}r:= g(w^* ){{< /katex >}}.
Suppose {{< katex >}}w^* {{< /katex >}} is _not_ a solution to {{< katex >}}(2){{< /katex >}}.
This means there is a {{< katex >}}w_0{{< /katex >}} such that {{< katex >}}g(w_0) \leq r{{< /katex >}} and {{< katex >}}J(w_0) < J(w^* ){{< /katex >}}.
But then 
{{< katex display>}}
    J(w_0) + \lambda g(w_0) 
        < J(w^* ) + \lambda g(w_0) 
        \leq J(w^* ) + \lambda r 
        = J(w^* ) + \lambda g(w^* )
{{< /katex >}}
which means {{< katex >}}w^* {{< /katex >}} is _not_ a solution to {{< katex >}}(1){{< /katex >}}, which is impossible.

### {{< katex >}}(2){{< /katex >}} implies {{< katex >}}(1){{< /katex >}}

Suppose now {{< katex >}}w^* {{< /katex >}} is a solution to {{< katex >}}(2){{< /katex >}} for a fixed {{< katex >}}r{{< /katex >}}.
We claim {{< katex >}}w^* {{< /katex >}} is also as solution to {{< katex >}}(2){{< /katex >}} for a fixed {{< katex >}}\lambda{{< /katex >}}.

We don't know just yet which {{< katex >}}\lambda{{< /katex >}} will do the trick, so let's pretend to have found it and see what happens.
Suppose {{< katex >}}w^* {{< /katex >}} is _not_ a solution to {{< katex >}}(1){{< /katex >}}.
This means there is a {{< katex >}}w_0{{< /katex >}} such that {{< katex >}}J(w_0) + \lambda g(w_0) < J(w^* ) + \lambda g(w^* ){{< /katex >}}.
Moreover, we must also have {{< katex >}}g(w^* ) < g(w_0){{< /katex >}}, otherwise
This means we must have {{< katex >}}\lambda < \frac {J(w^* ) - J(w_0)} {g(w^* ) - g(w_0)} {{< /katex >}}.
This suggests setting {{< katex >}}\lambda := \frac {J(w^* )}{ g(w^* ) } {{< /katex >}}.

Now that we have fixed {{< katex >}}\lambda{{< /katex >}}, suppose once more {{< katex >}}w^* {{< /katex >}} is _not_ a solution to {{< katex >}}(1){{< /katex >}}.
Then there exists {{< katex >}}w_0{{< /katex >}} such that {{< katex >}} J(w_0) + \lambda g(w_0) < J(w^* ) + \lambda g(w^* ) {{< /katex >}}.
If we had {{< katex >}}g(w_0) \leq r{{< /katex >}}, then {{< katex >}}J(w^* ) < J(w_0){{< /katex >}} as {{< katex >}}w^* {{< /katex >}} is a solution to {{< katex >}}(2){{< /katex >}}.
Therefore 
{{< katex display>}}
    J(w^* ) + \lambda g(w^* ) < J(w_0) + \lambda
{{< /katex >}}

##### Remark
Note that the proof goes through (for both directions) even when multiple solutions exist for either {{< katex >}}(1){{< /katex >}} or {{< katex >}}(2){{< /katex >}}.

{{< katex >}}{{< /katex >}}