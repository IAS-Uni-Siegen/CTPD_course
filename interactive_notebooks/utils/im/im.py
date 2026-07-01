"""Adapted from https://github.com/ExcitingSystems/exciting-environments.

The `L_m` saturation is based on the description in:

```
    @article{Stender2021,
    title={Accurate torque control for induction motors by utilizing a globally optimized flux observer},
    author={Stender, Marius and Wallscheid, Oliver and Boecker, Joachim},
    journal={IEEE Transactions on Power Electronics},
    volume={36},
    number={11},
    pages={13261--13274},
    year={2021},
    publisher={IEEE}
    }
```

"""

from copy import deepcopy
from dataclasses import fields
from functools import partial
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree import structure
import equinox as eqx
import diffrax
import jaxopt

from utils.common import clip_voltage_alpha_beta
from utils.im.motor_params import default_params


class IM(eqx.Module):
    T_s: float = eqx.field(static=True)
    _solver: diffrax.AbstractSolver = eqx.field(static=True)
    env_properties: eqx.Module
    action_dim: int = eqx.field(static=True)
    physical_state_dim: int = eqx.field(static=True)
    saturation_interpolators: dict

    def __init__(
        self,
        saturated: bool = False,
        static_params: dict = None,
        solver: diffrax.AbstractSolver = diffrax.Euler(),
        T_s: float = 1e-4,
    ):
        """
        Args:
            saturated (bool): If True, use lookup tables for saturated parameters. Default: False
            static_params (dict): Dictionary of static parameters for the IM.
                p (int): Pole pair number. Default: 2
                r_s (float): Stator resistance. Default: 2.9338
                r_r (float): Rotor resistance. Default: 1.355
                l_m (float): Main inductance. Default: 143.75e-3
                l_sigs (float): Stator-side stray inductance. Default: 5.87e-3
                l_sigr (float): Rotor-side stray inductance. Default: 5.87e-3
                u_dc (float): DC link voltage. Default: 560
                i_lim (float): Current limit. Default: 20
            solver (diffrax.solver): Solver for ODE integration. Default: diffrax.Euler()
            T_s (float): Duration of one control/simulation step in seconds. Default: 1e-4
        """
        motor_params = deepcopy(default_params(name=None))
        saturation_params = motor_params.saturation_params
        default_static_params = motor_params.static_params.__dict__

        if saturated:
            self.saturation_interpolators = self.generate_saturation_interpolators(
                saturation_params, motor_params.static_params
            )
        else:
            saturated_quants = [
                "l_m",
            ]
            self.saturation_interpolators = {q: lambda x: jnp.array([np.nan]) for q in saturated_quants}

        if static_params is None:
            static_params = default_static_params

        static_params = self.StaticParams(**static_params)
        env_properties = self.EnvProperties(
            saturated=saturated,
            static_params=static_params,
        )

        self.T_s = T_s
        self._solver = solver
        self.env_properties = env_properties
        self.action_dim = len(fields(self.Action))
        self.physical_state_dim = len(fields(self.PhysicalState))

    class State(eqx.Module):
        """The state of the environment."""

        physical_state: eqx.Module
        prng_key: jax.Array
        additions: eqx.Module

    class StaticParams(eqx.Module):
        """Static parameters of the IM."""

        p: jax.Array = eqx.field(converter=jnp.asarray)
        r_s: jax.Array = eqx.field(converter=jnp.asarray)
        r_r: jax.Array = eqx.field(converter=jnp.asarray)
        l_m: jax.Array = eqx.field(converter=jnp.asarray)
        l_sigs: jax.Array = eqx.field(converter=jnp.asarray)
        l_sigr: jax.Array = eqx.field(converter=jnp.asarray)
        u_dc: jax.Array = eqx.field(converter=jnp.asarray)
        i_lim: jax.Array = eqx.field(converter=jnp.asarray)

    class PhysicalState(eqx.Module):
        """Physical state of the IM."""

        epsilon: jax.Array  # Electrical rotor angle
        i_s_alpha: jax.Array  # Stator current, alpha
        i_s_beta: jax.Array  # Stator current, beta
        psi_r_alpha: jax.Array  # Rotor flux, alpha
        psi_r_beta: jax.Array  # Rotor flux, beta
        omega_el: jax.Array  # Electrical angular velocity
        torque: jax.Array  # Electromagnetic torque

    class Additions(eqx.Module):
        """Additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class Action(eqx.Module):
        """Action (voltage inputs) for the IM."""

        u_alpha: jax.Array
        u_beta: jax.Array

    class EnvProperties(eqx.Module):
        """Environment properties."""

        saturated: bool = eqx.field(static=True)
        static_params: eqx.Module

    def currents_to_torque(self, i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta):
        """Calculate electromagnetic torque for the IM."""
        params = self.env_properties.static_params
        l_r = params.l_sigr + params.l_m
        torque = 1.5 * params.p * params.l_m / l_r * (psi_r_alpha * i_s_beta - psi_r_beta * i_s_alpha)
        return torque

    def get_L_saturated(self, i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta):
        ax, ay = psi_r_alpha, psi_r_beta
        bx, by = i_s_alpha, i_s_beta
        psi_r_mag = jnp.sqrt(ax * ax + ay * ay)
        i_s_mag = jnp.sqrt(bx * bx + by * by)
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        angle = jnp.abs(jnp.atan2(cross, dot))

        query_point = jnp.stack([angle, psi_r_mag, i_s_mag])[None, :]
        l_m_sat = self.saturation_interpolators["l_m"](query_point)[0]
        return l_m_sat

    def currents_to_torque_sat(self, i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta):
        l_m_sat = self.get_L_saturated(i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta)
        params = self.env_properties.static_params
        l_r = params.l_sigr + l_m_sat
        torque = 1.5 * params.p * l_m_sat / l_r * (psi_r_alpha * i_s_beta - psi_r_beta * i_s_alpha)
        return torque

    def calc_flux_magnitudes(self, L_vec, i_sl_ab, psi_r_ab, l_sigr):
        lm_sat = L_vec
        i_r_ab = (psi_r_ab - lm_sat * i_sl_ab) / (lm_sat + l_sigr)

        psi_m_mag = jnp.linalg.norm(lm_sat * (i_sl_ab + i_r_ab))
        return psi_m_mag, i_r_ab

    def saturation_residuals_vector(self, L_vec, i_s_ab, psi_r_ab, params, static_params):
        lm_sat = L_vec
        psi_m_mag, i_r_ab = self.calc_flux_magnitudes(L_vec, i_s_ab, psi_r_ab, static_params.l_sigr)

        k1 = params.k1
        k2 = params.k2
        k3 = params.k3
        k4 = params.k4
        lm_new = k1 + (k1 - k2) / (1 + jnp.exp(-k3 * (0 - k4))) - (k1 - k2) / (1 + jnp.exp(-k3 * (psi_m_mag - k4)))

        return lm_sat - lm_new

    def generate_saturation_interpolators(self, nonlinear_params, static_params):

        i_s_max = static_params.i_lim
        psi_r_max = 0.8

        n_res = 50
        eps = 1e-6
        i_s_grid_1d = jnp.linspace(eps, i_s_max * 1.25, n_res)
        psi_r_grid_1d = jnp.linspace(eps, psi_r_max * 1.25, n_res)
        angle_grid_1d = jnp.linspace(0, jnp.pi, n_res)

        @jax.jit
        def solve_single_point(i_s_mag, psi_r_mag, angle, params):
            i_s_ab = jnp.array([i_s_mag, 0.0])
            psi_r_ab = jnp.array([psi_r_mag * jnp.cos(angle), psi_r_mag * jnp.sin(angle)])

            def root_fun(L_vec):
                return self.saturation_residuals_vector(L_vec, i_s_ab, psi_r_ab, params, static_params)

            solver = jaxopt.Broyden(fun=root_fun, maxiter=100, tol=1e-6)
            init_L = jnp.array(static_params.l_m)
            sol = solver.run(init_L)
            L_final = sol.params

            psi_m_mag, i_r_ab = self.calc_flux_magnitudes(L_final, i_s_ab, psi_r_ab, static_params.l_sigr)

            i_m_mag = jnp.linalg.norm(i_s_ab + i_r_ab)
            i_r_mag = jnp.linalg.norm(i_r_ab)

            return jnp.array([L_final, psi_m_mag, i_m_mag, i_r_mag, i_s_mag, psi_r_mag, angle])

        v_solve = jax.vmap(
            jax.vmap(jax.vmap(solve_single_point, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None)),
            in_axes=(None, None, 0, None),
        )
        grid_results = v_solve(i_s_grid_1d, psi_r_grid_1d, angle_grid_1d, nonlinear_params)

        lm_grid = grid_results[:, :, :, 0]

        points = (angle_grid_1d, psi_r_grid_1d, i_s_grid_1d)

        saturation_interpolators = {
            "l_m": jax.scipy.interpolate.RegularGridInterpolator(
                points, lm_grid, method="linear", bounds_error=False, fill_value=None
            ),
        }

        return saturation_interpolators

    def init_state(self):
        """Initialize the IM state."""
        env_properties = self.env_properties
        phys = self.PhysicalState(
            epsilon=jnp.array(0.0),
            i_s_alpha=jnp.array(0.0),
            i_s_beta=jnp.array(0.0),
            psi_r_alpha=jnp.array(0.0),
            psi_r_beta=jnp.array(0.0),
            omega_el=jnp.array(0.0),
            torque=jnp.array(0.0),
        )

        def voltage(t):
            return jnp.array([0.0, 0.0])

        args = (env_properties.static_params, phys.omega_el)
        if self.env_properties.saturated:
            vector_field = partial(self.nonlinear_ode, action=voltage)
        else:
            vector_field = partial(self.linear_ode, action=voltage)
        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.T_s
        y0 = (phys.i_s_alpha, phys.i_s_beta, phys.psi_r_alpha, phys.psi_r_beta, phys.epsilon)

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        return self.State(physical_state=phys, prng_key=jnp.array(jnp.nan), additions=additions)

    def linear_ode(self, t, y, args, action):
        """ODE for the IM in alpha-beta reference frame."""
        i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta, eps = y
        params, omega_el = args
        r_s = params.r_s
        r_r = params.r_r
        l_m = params.l_m
        l_r = params.l_sigr + l_m
        l_s = params.l_sigs + l_m
        sigma = (l_s * l_r - l_m**2) / (l_s * l_r)
        tau_r = l_r / r_r
        tau_sig = sigma * l_s / (r_s + r_r * (l_m**2) / (l_r**2))
        u_alpha_beta = action(t)
        u_alpha = u_alpha_beta[0]
        u_beta = u_alpha_beta[1]

        i_s_alpha_diff = (
            (-1 / tau_sig) * i_s_alpha
            + (l_m * r_r / (sigma * l_r**2 * l_s)) * psi_r_alpha
            + (l_m * omega_el / (sigma * l_r * l_s)) * psi_r_beta
            + (1 / (sigma * l_s)) * u_alpha
        )
        i_s_beta_diff = (
            (-1 / tau_sig) * i_s_beta
            + (-l_m * omega_el / (sigma * l_r * l_s)) * psi_r_alpha
            + (l_m * r_r / (sigma * l_r**2 * l_s)) * psi_r_beta
            + (1 / (sigma * l_s)) * u_beta
        )
        psi_r_alpha_diff = (l_m / tau_r) * i_s_alpha + (-1 / tau_r) * psi_r_alpha + (-omega_el) * psi_r_beta

        psi_r_beta_diff = (l_m / tau_r) * i_s_beta + (omega_el) * psi_r_alpha + (-1 / tau_r) * psi_r_beta

        eps_diff = omega_el
        d_y = i_s_alpha_diff, i_s_beta_diff, psi_r_alpha_diff, psi_r_beta_diff, eps_diff
        return d_y

    def nonlinear_ode(self, t, y, args, action):
        i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta, eps = y
        params, omega_el = args
        r_s = params.r_s
        r_r = params.r_r
        l_m = self.get_L_saturated(i_s_alpha, i_s_beta, psi_r_alpha, psi_r_beta)
        l_r = params.l_sigr + l_m
        l_s = params.l_sigs + l_m
        sigma = (l_s * l_r - l_m**2) / (l_s * l_r)
        tau_r = l_r / r_r
        tau_sig = sigma * l_s / (r_s + r_r * (l_m**2) / (l_r**2))
        u_alpha_beta = action(t)
        u_alpha = u_alpha_beta[0]
        u_beta = u_alpha_beta[1]

        i_s_alpha_diff = (
            (-1 / tau_sig) * i_s_alpha
            + (l_m * r_r / (sigma * l_r**2 * l_s)) * psi_r_alpha
            + (l_m * omega_el / (sigma * l_r * l_s)) * psi_r_beta
            + (1 / (sigma * l_s)) * u_alpha
        )
        i_s_beta_diff = (
            (-1 / tau_sig) * i_s_beta
            + (-l_m * omega_el / (sigma * l_r * l_s)) * psi_r_alpha
            + (l_m * r_r / (sigma * l_r**2 * l_s)) * psi_r_beta
            + (1 / (sigma * l_s)) * u_beta
        )
        psi_r_alpha_diff = (l_m / tau_r) * i_s_alpha + (-1 / tau_r) * psi_r_alpha + (-omega_el) * psi_r_beta

        psi_r_beta_diff = (l_m / tau_r) * i_s_beta + (omega_el) * psi_r_alpha + (-1 / tau_r) * psi_r_beta

        eps_diff = omega_el
        d_y = i_s_alpha_diff, i_s_beta_diff, psi_r_alpha_diff, psi_r_beta_diff, eps_diff
        return d_y

    @eqx.filter_jit
    def _ode_solver_step(self, state, u_alpha_beta):
        """Perform one ODE solver step."""
        properties = self.env_properties
        system_state = state.physical_state

        def voltage(t):
            return u_alpha_beta

        args = (properties.static_params, system_state.omega_el)
        if properties.saturated:
            vector_field = partial(self.nonlinear_ode, action=voltage)
        else:
            vector_field = partial(self.linear_ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.T_s
        y0 = (
            system_state.i_s_alpha,
            system_state.i_s_beta,
            system_state.psi_r_alpha,
            system_state.psi_r_beta,
            system_state.epsilon,
        )

        additions = jax.lax.cond(
            state.additions.active_solver_state,
            lambda _: self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True),
            lambda _: state.additions,
            operand=None,
        )

        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        i_s_alpha_k1, i_s_beta_k1, psi_r_alpha_k1, psi_r_beta_k1, eps_k1 = y
        eps_k1 = ((eps_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        if properties.saturated:
            torque = self.currents_to_torque_sat(i_s_alpha_k1, i_s_beta_k1, psi_r_alpha_k1, psi_r_beta_k1)
        else:
            torque = self.currents_to_torque(i_s_alpha_k1, i_s_beta_k1, psi_r_alpha_k1, psi_r_beta_k1)

        new_physical_state = eqx.tree_at(
            lambda s: (s.epsilon, s.i_s_alpha, s.i_s_beta, s.psi_r_alpha, s.psi_r_beta, s.torque),
            system_state,
            (eps_k1, i_s_alpha_k1, i_s_beta_k1, psi_r_alpha_k1, psi_r_beta_k1, torque),
        )
        new_additions = self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        new_state = eqx.tree_at(lambda s: (s.physical_state, s.additions), state, (new_physical_state, new_additions))
        return new_state

    def apply_voltage_constraint(self, u_alpha_beta, system_state):
        """Apply voltage constraints (hexagon clipping)."""
        return clip_voltage_alpha_beta(
            u_alpha_beta,
            self.env_properties.static_params.u_dc,
            system_state.physical_state.epsilon,
            system_state.physical_state.omega_el,
            self.T_s,
        )

    @eqx.filter_jit
    def step(self, state, action):
        """Perform one simulation step."""
        u_alpha_beta = self.apply_voltage_constraint(action, state)
        next_state = self._ode_solver_step(state, u_alpha_beta)
        observation = self.generate_observation(next_state)
        return observation, next_state

    def generate_observation(self, state):
        """Generate observation from state."""
        eps = state.physical_state.epsilon
        state_phys = state.physical_state
        obs = jnp.hstack(
            (
                state_phys.i_s_alpha,
                state_phys.i_s_beta,
                state_phys.omega_el,
            )
        )
        return obs

    def set_speed(self, n, state):
        """Set the mechanical speed (in rpm) and return new state and observation."""
        omega_el = jnp.array(self.env_properties.static_params.p * n * 2 * jnp.pi / 60)
        new_state = eqx.tree_at(lambda r: r.physical_state.omega_el, state, omega_el)
        obs = self.generate_observation(new_state)
        return obs, new_state

    @property
    def action_description(self):
        return ["u_alpha", "u_beta"]

    @property
    def obs_description(self):
        return np.array(["i_s_alpha", "i_s_beta", "omega_el"])

    def generate_state_from_observation(self, obs, key=None):
        """Generate state from observation."""
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan

        phys = self.PhysicalState(
            epsilon=jnp.arctan2(obs[7], obs[6]),
            i_s_alpha=obs[0],
            i_s_beta=obs[1],
            psi_r_alpha=obs[2],
            psi_r_beta=obs[3],
            torque=obs[5],
            omega_el=obs[4],
        )

        def voltage(t):
            return jnp.array([0, 0])

        args = (self.env_properties.static_params, phys.omega_el)
        if self.env_properties.saturated:
            vector_field = partial(self.nonlinear_ode, action=voltage)
        else:
            vector_field = partial(self.linear_ode, action=voltage)
        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.T_s
        y0 = (phys.i_s_alpha, phys.i_s_beta, phys.psi_r_alpha, phys.psi_r_beta, phys.epsilon)

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)
        state = self.State(physical_state=phys, prng_key=subkey, additions=additions)
        return state

    def reset(self):
        """Reset the environment to initial state."""
        state = self.init_state()
        obs = self.generate_observation(state)
        return obs, state
