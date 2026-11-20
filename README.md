# Module 4: Regularized Cost and Gradient (Solved)

Linear aur logistic regression ke cost function aur gradient me **L2 regularization** add kiya gaya hai.

## Files

| File | Description |
|------|-------------|
| `Module_4_Regularization_SOLVED.ipynb` | Solved notebook, code aur saved outputs ek hi jagah |
| `regularization_solved.py` | Same solution ka standalone Python script |

## Setup

```bash
pip install numpy matplotlib jupyter
```

```bash
jupyter notebook Module_4_Regularization_SOLVED.ipynb
# ya
python regularization_solved.py
```

## Solution summary

Regularization term: $\frac{\lambda}{2m}\sum_{j=0}^{n-1} w_j^2$. Parameter `b` regularize nahi hota (standard practice).

| Function | Regularization part |
|----------|---------------------|
| `compute_cost_linear_reg` | `reg_cost = (lambda_/(2*m)) * sum(w[j]**2)` |
| `compute_cost_logistic_reg` | `hx = sigmoid(w·x + b)`, cost `-y·log(hx) - (1-y)·log(1-hx)`, phir wahi `reg_cost` |
| `compute_gradient_linear_reg` | `dj_dw[j] += (lambda_/m) * w[j]` |
| `compute_gradient_logistic_reg` | `dj_dw[j] += (lambda_/m) * w[j]` |

Linear aur logistic ka fark sirf `h(x)` me hai: linear me `w·x + b`, logistic me `sigmoid(w·x + b)`.

## Results

| Check | Output |
|-------|--------|
| Linear regularized cost | 0.07917239320214277 (notebook), 0.07917239320214275 (script) |
| Logistic regularized cost | 0.6850849138741673 |
| Linear `dj_db` | 0.6648774569425726 |
| Linear `dj_dw` | [0.29653214748822276, 0.4911679625918033, 0.21645877535865857] |
| Logistic `dj_db` | 0.341798994972791 |
| Logistic `dj_dw` | [0.17380012933994293, 0.32007507881566943, 0.10776313396851499] |

## Notes
- Linear cost ka saved notebook output `0.07917239320214277` hai, jabki `regularization_solved.py` chalane par `0.07917239320214275` aaya (jo notebook ke Expected Output se match karta hai). Yeh sirf floating-point ke aakhri digit ka fark hai, jo NumPy version ya platform badalne par ho sakta hai. Result dono me sahi hai.
- Notebook me `matplotlib` import hai par is module me koi plot nahi hai.
