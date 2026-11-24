Single-server Markovian birth-death queue ($M/M/1$) with Poisson arrival rate $\lambda > 0$, exponential service rate $\mu > 0$,
and traffic utilization $\rho = \lambda / \mu < 1$, satisfying geometric stationary probabilities and Little's law:
$$\pi_k = (1 - \rho)\rho^k, \qquad L = \frac{\rho}{1 - \rho}, \qquad W = \frac{1}{\mu - \lambda}.$$

Reference: Gross, Shortle, Thompson, & Harris (2008), *Fundamentals of Queueing Theory*,
DOI: 10.1002/9781118625651; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Positive recurrence and finite stationary moments hold if and only if $\rho = \lambda / \mu < 1$; [`Self::new`] rejects $\rho \ge 1$ with [`QueueingError::UnstableQueue`].

# Domain of Validity
Requires strictly positive finite rates $\lambda > 0$, $\mu > 0$, stability margin $\lambda < \mu$, and positive state-space truncation capacity when constructing finite-state CTMCs.
