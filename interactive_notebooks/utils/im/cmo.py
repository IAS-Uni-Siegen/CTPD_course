import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx


class DiffraxCMO:
    R_r: float
    L_sigr: float
    L_m: float
    p: int
    T_s: float
    _solver: diffrax.AbstractSolver = eqx.field(static=True)

    class Additions(eqx.Module):
        solver_state: tuple
        active_solver_state: bool

    class CMOState(eqx.Module):
        eps_r_hat: jax.Array = eqx.field(converter=jnp.asarray)
        psi_r_hat: jax.Array = eqx.field(converter=jnp.asarray)
        omega_psi_rs_hat: jax.Array = eqx.field(converter=jnp.asarray)
        additions: eqx.Module

    def __init__(self, R_r, L_sigr, L_m, p, T_s, solver: diffrax.AbstractSolver = diffrax.Euler()):
        self.R_r = R_r
        self.L_sigr = L_sigr
        self.L_m = L_m
        self.p = p
        self.T_s = T_s
        self._solver = solver

    def reset(
        self,
        eps_r_hat=None,
        psi_r_hat=None,
        omega_psi_rs_hat=None,
    ):
        if eps_r_hat is None:
            eps_r_hat = 0.0
        if psi_r_hat is None:
            psi_r_hat = 0.0
        if omega_psi_rs_hat is None:
            omega_psi_rs_hat = 0.0

        return self.CMOState(
            eps_r_hat=eps_r_hat,
            psi_r_hat=psi_r_hat,
            omega_psi_rs_hat=omega_psi_rs_hat,
            additions=self.Additions(
                solver_state=self._solver.init(
                    diffrax.ODETerm(self._ode),
                    0.0,
                    self.T_s,
                    (eps_r_hat, psi_r_hat),
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ),
                active_solver_state=False,
            ),
        )

    def _ode(self, t, y, args):
        eps_r_hat, psi_r_hat = y

        # L_r, L_m, p, R_r, i_s_d, i_s_q, omega_rs = args
        L_r, L_m, p, R_r, i_s_alpha, i_s_beta, omega_rs = args

        # transform currents to est-dq
        cos_e, sin_e = jnp.cos(eps_r_hat), jnp.sin(eps_r_hat)
        i_s_d = i_s_alpha * cos_e + i_s_beta * sin_e
        i_s_q = -i_s_alpha * sin_e + i_s_beta * cos_e

        # estimate change in rotor flux magnitude
        d_psi_r = -self.R_r / L_r * psi_r_hat + self.R_r * self.L_m / L_r * i_s_d

        # estimate change in rotor flux angle
        omega_slip_hat = self.R_r * self.L_m / (L_r * (psi_r_hat + 1e-8)) * i_s_q
        omega_psi_rs_hat = omega_slip_hat + omega_rs
        d_eps_r = omega_psi_rs_hat

        dy = d_eps_r, d_psi_r
        return dy

    @eqx.filter_jit
    def __call__(
        self,
        state: CMOState,
        i_s_alpha: float,
        i_s_beta: float,
        omega_rs: float,
    ):
        L_r = self.L_sigr + self.L_m

        # # transform currents to est-dq
        # cos_e, sin_e = jnp.cos(state.eps_r_hat), jnp.sin(state.eps_r_hat)
        # i_s_d = i_s_alpha * cos_e + i_s_beta * sin_e
        # i_s_q = -i_s_alpha * sin_e + i_s_beta * cos_e

        # args = (L_r, self.L_m, self.p, self.R_r, i_s_d, i_s_q, omega_rs)
        args = (L_r, self.L_m, self.p, self.R_r, i_s_alpha, i_s_beta, omega_rs)

        term = diffrax.ODETerm(self._ode)

        t0 = 0.0
        t1 = self.T_s
        y0 = (state.eps_r_hat, state.psi_r_hat)

        additions = jax.lax.cond(
            state.additions.active_solver_state,
            lambda _: state.additions,
            lambda _: self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True),
            operand=None,
        )

        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)
        eps_r_hat_next, psi_r_hat_next = y
        eps_r_hat_next = ((eps_r_hat_next + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        i_s_q = -i_s_alpha * jnp.sin(eps_r_hat_next) + i_s_beta * jnp.cos(eps_r_hat_next)
        omega_slip_hat_next = self.R_r * self.L_m / (L_r * (psi_r_hat_next + 1e-8)) * i_s_q
        omega_psi_rs_hat_next = omega_slip_hat_next + omega_rs

        new_additions = self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        new_state = self.CMOState(
            psi_r_hat=psi_r_hat_next,
            eps_r_hat=eps_r_hat_next,
            omega_psi_rs_hat=omega_psi_rs_hat_next,
            additions=new_additions,
        )
        return new_state
