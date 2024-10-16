import jax
import jax.numpy as jnp
from flax import nnx

###########################################################################################################
# Helper Functions                                                                                        #
###########################################################################################################

# Function for batched matrix-vector multiplication
def batch_mat_vec_mul(M, x):
    return jax.lax.dot_general(M, x, dimension_numbers=((2, 1), (0, 0)))

# Identity activation function
@jax.jit
def identity(x):
    return x

###########################################################################################################
# Model                                                                                                   #
###########################################################################################################

class JaxEKF(nnx.Module):
    def __init__(self, input_dim, state_dim, rngs):
        self.input_dim = input_dim
        self.state_dim = state_dim

        self.f = identity
        self.h = identity

        # self.f = nnx.Sequential(
        #     #nnx.Linear(self.state_dim, self.state_dim, rngs=rngs),
        #     #nnx.sigmoid,
        #     #nnx.Linear(self.state_dim, self.state_dim, rngs=rngs)
        # )
        # self.h = nnx.Sequential(
        #     #nnx.Linear(self.state_dim, self.input_dim, rngs=rngs),
        #     #nnx.sigmoid,
        #     #nnx.Linear(self.input_dim, self.input_dim, rngs=rngs)
        # )

        self.Q = nnx.Param(jnp.eye(self.state_dim) * 0.1)
        self.R = nnx.Param(jnp.eye(self.input_dim) * 0.1)

###########################################################################################################
# EKF - Sequential                                                                                        #
###########################################################################################################
    # ekf sequential forward
    def forward_seq(self, x):
        X_k = jnp.zeros((x.shape[1], self.state_dim))                                               # shape=(batch, state_dim)
        P_k = jnp.repeat(jnp.eye(self.state_dim, self.state_dim)[None, :, :], x.shape[1], axis=0)   # repeat P0 in batch dimension, shape=(batch, state_dim, state_dim)
        X_ks = jnp.zeros((x.shape[0], x.shape[1], self.state_dim))                                  # shape=(sequence length, batch, state_dim)

        for i in range(x.shape[0]):
            X_k_pred, P_k_pred = self.predict(X_k, P_k)
            X_k, P_k = self.update(X_k_pred, P_k_pred, x.at[i].get())
            X_ks = X_ks.at[i].set(X_k)

        return X_ks
    
    # ekf prediction step
    def predict(self, X_k, P_k):
        X_k_pred = self.f(X_k)
        F = jax.vmap(jax.jacobian(self.f))(X_k)

        P_k_pred = jnp.matmul(jnp.matmul(F, P_k), F.mT) + self.Q

        return X_k_pred, P_k_pred

    # ekf update step
    def update(self, X_k_pred, P_k_pred, Z_k):
        y_k = (Z_k - self.h(X_k_pred))
        H = jax.vmap(jax.jacobian(self.h))(X_k_pred)

        S_k = jnp.matmul(jnp.matmul(H, P_k_pred), H.mT) + self.R
        K_k = jnp.matmul(jnp.matmul(P_k_pred, H.mT), jnp.linalg.inv(S_k))

        X_k = X_k_pred + batch_mat_vec_mul(K_k, y_k)
        P_k = P_k_pred - jnp.matmul(jnp.matmul(K_k, H), P_k_pred)

        return X_k, P_k
    
###########################################################################################################
# EKF - Scan                                                                                              #
###########################################################################################################
    # ekf scan forward
    def forward_scan(self, x):
        X_k = jnp.zeros((x.shape[1], self.state_dim))                                               # shape=(batch, state_dim)
        P_k = jnp.repeat(jnp.eye(self.state_dim, self.state_dim)[None, :, :], x.shape[1], axis=0)   # shape=(batch, state_dim, state_dim)

        @jax.vmap
        def params(obs, i, x_k_1, x_k):
            return self.make_params(obs, i, x_k_1, x_k, X_k, P_k)
        
        X_ks = X_ks_1 = jnp.zeros((x.shape[0], x.shape[1], self.state_dim))                         # shape=(sequence length, batch, state_dim)

        As, bs, Cs, etas, Js = params(x, jnp.arange(x.shape[0]), X_ks_1, X_ks)
        _, filtered_means, filtered_covariances, _, _ = jax.lax.associative_scan(jax.vmap(jax.vmap(self.filtering_operator)), (As, bs, Cs, etas, Js))

        return filtered_means

    # init ekf scan parameters
    def make_params(self, Z_k, i, x_k_1, x_k, m0, P0):
        predicate = i == 0

        def first():
            return self.make_first(m0, P0, x_k_1, x_k, Z_k, True)
        
        def generic():
            return self.make_generic(x_k_1, x_k, P0, Z_k)

        return jax.lax.cond(
            predicate,
            first,
            generic,
        )
    
    # for first time step
    def make_first(self, m0, P0, x_k_1, x_k, y, propagate_first):
        if propagate_first:
            F = jax.vmap(jax.jacobian(self.f))(x_k_1)
            m = batch_mat_vec_mul(F, m0 - x_k_1) + self.f(x_k_1)
            P = F @ P0 @ F.mT + self.Q
            H = jax.vmap(jax.jacobian(self.h))(x_k)
            alpha = self.h(x_k) + batch_mat_vec_mul(H, m - x_k)
        else:
            P = P0
            m = m0
            H = jax.vmap(jax.jacobian(self.h))(x_k_1)
            alpha = self.h(x_k_1) + batch_mat_vec_mul(H, m0 - x_k_1)

        S = H @ P @ H.mT + self.R
        K = jnp.linalg.solve(S, H @ P).mT
        A = jnp.zeros_like(P0)

        b = m + batch_mat_vec_mul(K, y - alpha)
        C = P - (K @ S @ K.mT)

        eta = jnp.zeros_like(m0)
        J = jnp.zeros_like(P0)

        return A, b, C, eta, J

    # for generic time steps
    def make_generic(self, x_k_1, x_k, Qk_1, yk):
        F = jax.vmap(jax.jacobian(self.f))(x_k_1)
        H = jax.vmap(jax.jacobian(self.h))(x_k)

        F_x_k_1 = batch_mat_vec_mul(F, x_k_1)
        x_k_hat = self.f(x_k_1)

        alpha = self.h(x_k) + batch_mat_vec_mul(H, x_k_hat - F_x_k_1 - x_k)
        residual = yk - alpha
        HQ = H @ Qk_1

        S = HQ @ H.mT + self.R
        S_invH = jnp.linalg.solve(S, H)
        K = (S_invH @ Qk_1).mT
        A = F - K @ H @ F
        b = batch_mat_vec_mul(K, residual) + x_k_hat - F_x_k_1
        C = Qk_1 - K @ H @ Qk_1

        HF = H @ F

        temp = (S_invH @ F).mT
        eta = batch_mat_vec_mul(temp, residual)
        J = temp @ HF

        return A, b, C, eta, J
    
    # ekf scan operator
    def filtering_operator(self, elem1, elem2):
        A1, b1, C1, eta1, J1 = elem1
        A2, b2, C2, eta2, J2 = elem2
        dim = b1.shape[0]

        I_dim = jnp.eye(dim)

        IpCJ = I_dim + jnp.dot(C1, J2)
        IpJC = I_dim + jnp.dot(J2, C1)

        AIpCJ_inv = jnp.linalg.solve(IpCJ.mT, A2.mT).mT
        AIpJC_inv = jnp.linalg.solve(IpJC.mT, A1).mT

        A = jnp.dot(AIpCJ_inv, A1)
        b = jnp.dot(AIpCJ_inv, b1 + jnp.dot(C1, eta2)) + b2
        C = jnp.dot(AIpCJ_inv, jnp.dot(C1, A2.mT)) + C2
        eta = jnp.dot(AIpJC_inv, eta2 - jnp.dot(J2, b1)) + eta1
        J = jnp.dot(AIpJC_inv, jnp.dot(J2, A1)) + J1
        return A, b, 0.5 * (C + C.mT), eta, 0.5 * (J + J.mT)

###########################################################################################################
# Main                                                                                                    #
###########################################################################################################

if __name__ == "__main__":
    rngs = nnx.Rngs(jax.random.PRNGKey(42))

    # Create a random input
    x = jax.random.uniform(rngs.params(), (128, 8, 16)) # shape=(sequence length, batch, state_dim)

    # Init nnx model
    test = JaxEKF(input_dim=16, state_dim=16, rngs=rngs)
    nnx.display(test)

    # Sequential forward
    X_ks_seq = test.forward_seq(x)
    print(X_ks_seq.shape)

    # Scan forward
    X_ks_scan = test.forward_scan(x)
    print(X_ks_scan.shape)

    # Compare states
    print("Differences")
    for i in range(X_ks_seq.shape[0]):
        print(f"t={i}: mean: {jnp.abs(X_ks_seq[i] - X_ks_scan[i]).mean():10e}\t max: {jnp.abs(X_ks_seq[i] - X_ks_scan[i]).max():10e}")

    print("Done")