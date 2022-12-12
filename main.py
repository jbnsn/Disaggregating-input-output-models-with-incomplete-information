"""Implementation of Lindner et al. (2012) in Python with NumPy and Pandas.

Lindner, Sören, Julien Legault, and Dabo Guan. 2012.
‘Disaggregating Input–Output Models with Incomplete Information’.
Economic Systems Research 24 (4): 329–47.
https://doi.org/10.1080/09535314.2012.689954.

The comments in this script are the same as the ones in the Matlab code
in the supplementary material 'cesr_a_689954_sup_27358897.docx'
of Lindner (2012).

Source (accessed 06.12.2022):
https://www.tandfonline.com/doi/suppl/10.1080/09535314.2012.689954

The script contains the generation of random numbers. A random vector is
generated in line 90 of the Matlab script:

    `base(p,:) = rand(1,Nv)`

For verification purposes, `np.random.seed(1337)` (Python) and
`rand('twister', 1337)` (Matlab) was applied.

"""

import numpy as np
import pandas as pd

from tqdm import tqdm

if False:  # !!!

    # Switch flag for verification
    # Matlab equivalent: `rand('twister', 1337)`
    # Source: https://stackoverflow.com/a/20202330/5696601

    np.random.seed(1337)

# %% Loading IO data

flows = pd.read_csv(
    # Input–output table of China (2007), in billion RMB
    './input/io-table-cn-2007-flows.csv',
    header=None
    )

flows_idx = pd.read_csv(
    './input/io-table-cn-2007-flows-idx.csv'
    )

flows.columns = pd.MultiIndex.from_frame(flows_idx)
flows.index = pd.MultiIndex.from_frame(flows_idx.iloc[:12, :])

# Vector of final demand
f = flows.loc[:, ('Final demand', 'FD')]

# Vector of intermediate demand
id = flows.loc[:, ('Intermediate demand', 'ID')]

# Vector of total outputs
x = f + id

# Exchange matrix
Z = flows.loc[
    # Rows
    :,
    # Cols
    (~flows.columns.get_level_values('Cat')
     .isin(['ID', 'FD', 'TO']))
    ]

del flows_idx

temp = Z.shape  # Size of IO table

N = temp[0] - 1  # Number of common sectors

A = np.divide(Z, x)  # ggregated technical coefficient matrix

x_common = x[:-1]  # Vector of total outputs for common sectors

f_common = f[:-1]  # Vector of final demand for common sectors

# Note: The last sector of the table is disaggregated,
# i.e. the electricity sector

x_elec = x[-1]  # Total output of the disaggregated sector

f_elec = f[-1]  # Final demand of the disaggregated sector

# %% Newly formed sectors from the electricity sector

# New sector weights
w = pd.read_csv(
    './input/io-table-cn-2007-w.csv',
    header=None
    )

w = w.values.flatten()

w_idx = pd.read_csv(
    './input/io-table-cn-2007-w-idx.csv'
    )

n = len(w)  # Number of new sectors

# Total number of sectors for the disaggregated IO table
N_tot = N + n

# Vector of new total sector outputs
x_new = w*x_elec/1000

# Vector of disaggregated economy sector total outputs
xs = np.concatenate((x_common, x_new))

f_new = w*f_elec  # # Final demand of new sectors

# %% Building the constraint matrix C

Nv = n * N_tot + n  # Number of variables

Nc = N + n + 1  # Number of constraints

# Vector of constraint constants
q = pd.concat(
    [A.iloc[N, :],
     pd.Series(w, index=pd.MultiIndex.from_frame(w_idx))]
    )

# Matrix of constraints
C = np.zeros((Nc, Nv))

# %%% Common sectors constraints

C11 = np.zeros((N, N*n))

for ii in range(N):
    col_indices = range(n*(ii), n*ii+n)
    C11[ii, col_indices] = np.ones((1, n))

C[:N, :N*n] = C11

# %%% New sectors constraints

C22 = np.zeros((1, n**2))

for ii in range(0, n):
    col_indices = range(n*(ii), n*ii+n)
    C22[0, col_indices] = w[ii]*np.ones((1, n))

C[N, N*n:N*n+n**2] = C22

# %%% Final demand constraints

C31 = np.zeros((n, N*n))

for ii in range(N):
    col_indices = range(n*(ii-1)+3, n*ii+3)
    C31[:n, col_indices] = (x_common[ii]/x_elec)*np.eye(n)

C32 = np.zeros((n, n**2))

for ii in range(0, n):
    col_indices = range(n*(ii-1)+3, n*ii+3)
    C32[:n, col_indices] = w[ii]*np.eye(n)

C[N+1:, :N*n] = C31
C[N+1:, N*n:N*n+n**2] = C32
C[N+1:, N*n+n**2:] = np.eye(n)

# %% Building the initial estimate y0

# Technical coefficient matrix of the initial estimate
As_y0 = np.zeros((N_tot, N_tot))

# Common/Common part
As_y0[:N, :N] = A.iloc[:N, :N]

# Common/New part
As_y0[:N, N:N_tot] = np.repeat(A.iloc[:N, N].to_numpy(), n).reshape(N, n)

# New/Common part
As_y0[N:N_tot, :N] = (
    np.multiply(w, A.iloc[N, :N].to_numpy().repeat(n).reshape(N, n)).T
    )

# New/New part
As_y0[N:N_tot, N:N_tot] = np.multiply(
    A.iloc[N, N],
    np.repeat(w, n).reshape(n, n)
    )

# %% Generating the orthogonal distinguishing matrix

# %%% Making the constraint matrix orthogonal

C_orth = C.copy()

for c in tqdm(range(Nc), desc='Orthogonalize constraint matrix'):
    for i in range(c):

        # Orthogonal projection
        C_orth[c, :] = (
            C_orth[c, :]
            - np.dot(C_orth[c, :], C_orth[i, :])
            / np.linalg.norm(C_orth[i, :])**2 * C_orth[i, :]
            )

# %%% Gram-Schmidt algorithm

base = np.zeros((Nv, Nv))  # Orthogonal base containing C_orth and D
base[:Nc, :] = C_orth.copy()

for p in tqdm(range(Nc, Nv), desc='Gram-Schmidt algorithm'):

    # Generate random vector
    base[p, :] = np.random.rand(1, Nv)

    for i in range(p):

        # Orthogonal projection on previous vectors
        base[p, :] -= (
            np.dot(base[p, :], base[i, :])
            / np.linalg.norm(base[i, :])**2
            * base[i, :]
            )

    # Normalizing
    base[p, :] /= np.linalg.norm(base[p, :])

# Retrieving the distinguishing matrix from the orthogonal base
D = base[Nc:, :].T
