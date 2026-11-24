Multi-server Markovian birth-death queue ($M/M/c$ Erlang-C model) with arrival rate $\lambda > 0$, per-server rate $\mu > 0$,
$c \ge 1$ identical parallel servers, offered load $a = \lambda / \mu$, and per-server utilization $\rho = \frac{\lambda}{c\mu} < 1$:
$$C(c, a) = \frac{\frac{a^c}{c!(1 - \rho)}}{\sum_{k=0}^{c-1}\frac{a^k}{k!} + \frac{a^c}{c!(1 - \rho)}}, \qquad L_q = \frac{\rho\,C(c, a)}{1 - \rho}, \qquad W = W_q + \frac{1}{\mu}.$$

Reference: Erlang (1917) & Gross et al. (2008), *Fundamentals of Queueing Theory*,
DOI: 10.1002/9781118625651; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Evaluates $a^k / k!$ recurrences incrementally from term $k = 0$ to avoid factorial overflow at large server counts $c$, enforcing ergodicity via $\rho = \lambda / (c\mu) < 1$ (`QueueingError::UnstableQueue`).

# Domain of Validity
Requires finite rates $\lambda > 0$, $\mu > 0$, server count $c \ge 1$ (`QueueingError::InvalidServerCount`), and sub-unity utilization $\lambda < c\mu$.
