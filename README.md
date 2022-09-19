# k-greedy-availability-aware

Availability-aware role-mining wih k-Greedy algorithm 

Based on k-Greedy algorithm implementation from:
https://github.com/kovacsrekaagnes/rank_k_Binary_Matrix_Factorisation

### Motivation
Role-mining algorithms with Frobenius norm usually tends to security-aware factorisations due to sparsity of the initial matrix.

### Usage
```
import ..BMF
number_of_roles = 70
numner_of_addittion_iteration = 140
W, H = BMF.BMF_k_greedy_heur(X, number_of_roles, add = numner_of_addittion_iteration)
```
