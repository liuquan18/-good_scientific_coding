we seek anomaly patterns associated with time series $\vectorbold t_k$ that maximize the ratio of (ensemble mean) signal to total variance:
$$
s_k = \frac {\langle \vectorbold t_k \rangle^T \langle \vectorbold t_k \rang} {\vectorbold t_k^T \vectorbold t_k}\label {ref1}\tag{1}
$$
Here, angle brackets denote an ensemble average. These time series are determined by the projection of a fingerprint pattern $\vectorbold u_k$ onto the semble data matrix $\mathbf X$:
$$
\vectorbold t_k = \vectorbold X\vectorbold u_k\label{ref2}\tag{2}
$$
The $n \cdot n_e \times p$  ensemble data matrix $\mathbf X$ is constructed by concatenating the $n\times p$ data matrice $\mathbf X_i$ from each ensemble member in the time dimension, where $n$ is the length of time series, $n_e$ is the number of ensemble members, and $p$ is the spatial dimension. Each ensemble member data matrix $\mathbf X_i$ is weighted by the square root of grid cell area, such that the covariance matrix is area weighted.

To ensure that the identified patterns corespond to variability that actually occurs within the ensemble, the fingerprint patterns $\mathbf u_k$ are required to be linear combinations of the $N$ leading ensemble $EOFs$ $\mathbf a_k$, with normalized weight vectors $\mathbf e_k$:
$$
\mathbf u_k = \begin {bmatrix}\mathbf a_1 \over \sigma _1 & \mathbf a_2 \over \sigma _2 & \cdots & \mathbf a_N \over \sigma _N \end{bmatrix} \mathbf e_k \label{ref3}\tag{3}
$$
The ensemble $EOFs$ $\mathbf a_k$ are eigenvectors of the ensemble-mean covariance matrix $\lang \mathbf C \rang$ ,
$$
\lang \mathbf C \rang \mathbf a_k = \sigma_k^2 \mathbf a_k \label{ref4}\tag4
$$


Where
$$
\lang \mathbf C\rang = n_E^{-1} \sum_{i=1}^{n_E}\mathbf C_i \label{ref5}\tag{5}
$$