# Metrics

### 1. **Policy Divergence**
```python
def policy_divergence(T_old, T_new, epsilon=1e-10):
    kl1 = T_old * np.log((T_old + epsilon)/(T_new + epsilon))
    kl2 = T_new * np.log((T_new + epsilon)/(T_old + epsilon))
    return 0.5*(kl1.sum(axis=(1,2)) + kl2.sum(axis=(1,2)))
```
- Quantifies the bidirectional divergence between successive policy versions using a symmetric Kullback-Leibler formulation:
  $$
  D_{sym}(P\|Q) = \frac{1}{2}[D_{KL}(P\|Q) + D_{KL}(Q\|P)]
  $$
- Tracks policy stability during learning. High values indicate significant policy changes between updates.

### 2. **Action Policy Entropy**
```python
def action_policy_entropy(T):
    return -np.sum(T * np.log2(T + 1e-10), axis=(1,2))
```
- Shannon entropy of the policy's action distribution:
  $$
  H(T) = -\sum_{a,s'} T(a,s'|s) \log_2 T(a,s'|s)
  $$
- Monitors exploration/exploitation balance:
  - High entropy: Exploratory behavior (uniform action distribution)
  - Low entropy: Exploitative behavior (peaked distribution)
- Uses base-2 logarithm for interpretability in bits

### 3. **Effective Clones**
```python
def effective_clones(C, threshold=1e-3):
    return np.sum(np.max(C, axis=0) > threshold, axis=1)
```
- Number of active clones per state that exceed an activation threshold
  - Induces capacity bottleneck for efficient learning
  - Prevents overfitting by limiting clone amount
  - Measures model complexity in CSCG framework

### 4. **Stationary Distribution Analysis**
```python
def stationary_distribution(T):
    evals, evecs = np.linalg.eig(T.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    return evec1 / evec1.sum()
```
- Long-term visitation distribution π satisfying:
  $$
  \pi = \pi T
  $$
- Reveals absorbing states and policy dead-ends
- Helps identify policy convergence properties
- Used for off-policy evaluation in RL

### 5. **Volatility Index**
```python
# In update()
if len(metrics['divergence']) > 10:
    recent_div = metrics['divergence'][-10:]
    metrics['volatility'].append(np.std(recent_div))
```
- Rolling standard deviation of policy divergences over 10 updates
- Detects oscillatory learning dynamics
- Early indicator of convergence (low volatility)
- Helps tune learning rates adaptively
