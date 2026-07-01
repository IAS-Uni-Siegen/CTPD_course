import jax
import jax.numpy as jnp

import equinox as eqx

from utils.pmsm.pmsm import PMSM, clip_voltage, lut_interpolate


class FOCController:

    def __init__(
        self,
        static_params: PMSM.StaticParams,
        lut_grids: jax.Array,
        lut_values: jax.Array,
        T_s: float,
        d: float,
        rho: float,
        use_ARW: bool = False,
        K_r: float = 1.0,
    ):
        """Gathers all static parameters for the controller and prepares them for later usage

        Args:
            static_params (PMSM.StaticParams): The static parameters of the machine, e.g., stator resistance
            lut_grids (jax.Array): Support grids for the LUTs
            lut_values (jax.Array): Values for the LUTs
            T_s (float): Sampling time for the simulation
            d (float): Damping factor that defines the shape of the transient response (value in [0, 1])
            rho (float): Transient factor that defines the settling time (value > 0)
            use_ARW (bool): Whether anti-reset wind-up should be used
            K_r (float): Anti-reset wind-up reference gain
        """
        self.T_s = T_s
        self.static_params = static_params
        self.lut_grids = lut_grids
        self.lut_values = lut_values
        self.d = d
        self.rho = rho

        # Equation (3.88)
        self.kappa = jnp.sqrt(1 - self.d**2) / d

        # Equation (3.94)
        self.P_lambda = jnp.exp(-2 * self.rho)
        self.S_lambda = 2 * jnp.exp(-self.rho) * jnp.cos(self.kappa * self.rho)

        self.z_r = jnp.exp(-self.rho) * jnp.cos(self.kappa * self.rho)
        self.use_ARW = use_ARW
        if self.use_ARW:
            self.K_r = K_r

    def park_transformation_matrix(self, eps):
        cos = jnp.cos(eps)
        sin = jnp.sin(eps)
        T_p = jnp.column_stack((cos, -sin, sin, cos)).reshape(2, 2)
        return T_p

    def get_parameters_for_op(self, i_dq):
        """Extracts the momentary value for the current-dependent quantities from the LUTs."""

        p_d = {q: lut_interpolate(*self.lut_grids[q], self.lut_values[q], *i_dq) for q in self.lut_grids}
        L_diff = jnp.column_stack([p_d[q] for q in ["L_dd", "L_dq", "L_qd", "L_qq"]]).reshape(2, 2)
        psi_dq = jnp.column_stack([p_d[q] for q in ["Psi_d", "Psi_q"]])
        return L_diff, jnp.squeeze(psi_dq)

    def predict_currents(self, i_dq, u_dq, omega):
        """Predicts the currents in the next sampling period based on Equation (3.56).

        Args:
            i_dq (jax.Array): Momentary current operation point
            u_dq (jax.Array): Momentary applied voltage
            omega (jax.Array): Electrical angular velocity

        Returns:
            Current prediction for the next sampling period
        """
        L_diff, psi_dq = self.get_parameters_for_op(i_dq)
        L_diff_inv = jnp.linalg.inv(L_diff)

        T_p_k = self.park_transformation_matrix(-omega * self.T_s)

        i_dq_pred = (
            (jnp.eye(2) - L_diff_inv @ T_p_k * self.T_s * self.static_params.r_s) @ i_dq
            + L_diff_inv @ (T_p_k - jnp.eye(2)) @ psi_dq
            + L_diff_inv @ T_p_k * self.T_s @ u_dq
        )

        return i_dq_pred

    def compute_gains(self, L_diff):
        """Recomputes the controller gains based on the momentary differential inductance matrix.

        Uses Equations (3.71) for a and b, then Equation (3.95).
        """
        L_dd = L_diff[0, 0]
        L_qq = L_diff[1, 1]

        a = jnp.array(
            [
                1 - (self.T_s * self.static_params.r_s) / L_dd,
                1 - (self.T_s * self.static_params.r_s) / L_qq,
            ]
        )

        b = jnp.array(
            [
                self.T_s / L_dd,
                self.T_s / L_qq,
            ]
        )

        K_p = (a - self.P_lambda) / b
        K_i = (1 - self.S_lambda + self.P_lambda) / (b * self.T_s)

        K_f = (self.z_r * K_i * self.T_s) / (1 - self.z_r) - K_p

        return K_p, K_i, K_f

    def integrate(self, integrated, e):
        """Error integration based on Equation (3.74)."""
        integrated = integrated + e * self.T_s
        return integrated

    def decouple(self, i_dq, v, L_diff, psi_dq, omega):
        """Decoupling based on Equation (3.69)."""
        T_p_k = self.park_transformation_matrix(-omega * self.T_s)
        T_p_k_inv = jnp.linalg.inv(T_p_k)

        D_inv = jnp.array([[1 / L_diff[0, 0], 0], [0, 1 / L_diff[1, 1]]])
        u_dq = (
            (T_p_k_inv - jnp.eye(2)) / self.T_s @ psi_dq
            + (jnp.eye(2) - T_p_k_inv @ L_diff @ D_inv) * self.static_params.r_s @ i_dq
            + T_p_k_inv @ L_diff @ D_inv @ v
        )
        return u_dq

    def reset(self):
        return jnp.zeros((2,))

    def correct_integration(self, integrated, delta_u_dq, K_i, K_p):
        """Corrects integration for anti-reset wind-up based on Equations (3.118) and (3.113)."""
        K_aw = K_i * self.T_s / (K_p + K_i * self.T_s) * self.K_r

        return integrated - K_aw / K_i * delta_u_dq

    @eqx.filter_jit
    def __call__(self, i_dq, i_dq_ref, u_dq_appl, omega, eps, integrated):
        """Chooses the voltage command.

        Args:
            i_dq (jax.Array): Momentary current operating point
            i_dq_ref (jax.Array): Reference current
            u_dq_appl (jax.Array): Voltage that is applied in the momentary sampling period
            omega (jax.Array): Electrical angular velocity
            eps (jax.Array): Electrical angle
            integrated (jax.Array): Integrator state

        Returns:
            Tuple of voltage command and updated integrator state
        """
        i_dq_pred = self.predict_currents(i_dq, u_dq_appl, omega)
        e = i_dq_ref - i_dq_pred

        integrated = self.integrate(integrated, e)
        L_diff, psi_dq = self.get_parameters_for_op(i_dq_pred)
        K_p, K_i, K_f = self.compute_gains(L_diff)

        # Equations (3.75), (3.76), and (3.73)
        v = K_p * e + K_i * integrated + K_f * i_dq_ref

        u_dq = self.decouple(i_dq_pred, v, L_diff, psi_dq, omega)
        u_dq_clipped = clip_voltage(u_dq, self.static_params.u_dc, eps, omega, self.T_s)

        if self.use_ARW:
            delta_u_dq = u_dq - u_dq_clipped
            integrated = self.correct_integration(integrated, delta_u_dq, K_i, K_p)

        return u_dq_clipped, integrated
