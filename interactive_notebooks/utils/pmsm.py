"""Adapted from https://github.com/ExcitingSystems/exciting-environments."""

from dataclasses import fields
from functools import partial
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree import structure
import equinox as eqx
import diffrax

from utils.motor_params import MotorVariant


# only for alpha/beta -> abc
t32 = jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]])
t23 = 2 / 3 * jnp.array([[1, 0], [-0.5, 0.5 * jnp.sqrt(3)], [-0.5, -0.5 * jnp.sqrt(3)]]).T  # only for abc -> alpha/beta

inverter_t_abc = jnp.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ]
)

ROTATION_MAP = np.ones((2, 2, 2), dtype=np.complex64)
ROTATION_MAP[1, 0, 1] = 0.5 * (1 + np.sqrt(3) * 1j)
ROTATION_MAP[1, 1, 0] = 0.5 * (1 - np.sqrt(3) * 1j)
ROTATION_MAP[0, 1, 0] = 0.5 * (-1 - np.sqrt(3) * 1j)
ROTATION_MAP[0, 1, 1] = -1
ROTATION_MAP[0, 0, 1] = 0.5 * (-1 + np.sqrt(3) * 1j)
ROTATION_MAP = jnp.array(ROTATION_MAP)


def t_dq_alpha_beta(eps):
    """Compute the transformation matrix for converting between DQ and Alpha-Beta reference frames."""
    cos = jnp.cos(eps)
    sin = jnp.sin(eps)
    return jnp.column_stack((cos, sin, -sin, cos)).reshape(2, 2)


def dq2abc(u_dq, eps):
    """Transform voltages from DQ coordinates to ABC (three-phase) coordinates."""
    u_abc = t32 @ dq2albet(u_dq, eps).T
    return u_abc.T


def dq2albet(u_dq, eps):
    """Transform voltages from DQ coordinates to Alpha-Beta coordinates."""
    q = t_dq_alpha_beta(-eps)
    u_alpha_beta = q @ u_dq.T

    return u_alpha_beta.T


def albet2dq(u_albet, eps):
    """Transform voltages from Alpha-Beta coordinates to DQ coordinates."""
    q_inv = t_dq_alpha_beta(eps)
    u_dq = q_inv @ u_albet.T

    return u_dq.T


def abc2dq(u_abc, eps):
    """Transform voltages from ABC (three-phase) coordinates to DQ coordinates."""
    u_alpha_beta = t23 @ u_abc.T
    u_dq = albet2dq(u_alpha_beta.T, eps)
    return u_dq


def step_eps(eps, omega_el, T_s, T_s_scale=1.0):
    """Update the electrical angle over a time step with optional scaling."""
    eps += omega_el * T_s * T_s_scale
    eps %= 2 * jnp.pi
    boolean = eps > jnp.pi
    summation_mask = boolean * -2 * jnp.pi
    eps = eps + summation_mask
    return eps


def apply_hex_constraint(u_albet):
    """Clip voltages in alpha/beta coordinates into the voltage hexagon."""
    u_albet_c = u_albet[0] + 1j * u_albet[1]
    idx = (jnp.sin(jnp.angle(u_albet_c)[..., jnp.newaxis] - 2 / 3 * jnp.pi * jnp.arange(3)) >= 0).astype(int)
    rot_vec = ROTATION_MAP[idx[0], idx[1], idx[2]]
    # rotate sectors upwards
    u_albet_c = jnp.multiply(u_albet_c, rot_vec)
    u_albet_c = jnp.clip(u_albet_c.real, -2 / 3, 2 / 3) + 1j * u_albet_c.imag
    u_albet_c = u_albet_c.real + 1j * jnp.clip(u_albet_c.imag, 0, 2 / 3 * jnp.sqrt(3))
    u_albet_c = jnp.multiply(u_albet_c, jnp.conjugate(rot_vec))  # rotate back
    return jnp.column_stack([u_albet_c.real, u_albet_c.imag])


def clip_in_abc_coordinates(u_dq, u_dc, omega_el, eps, T_s):
    """Clip voltages in ABC (three-phase) coordinates and transform back to DQ coordinates."""
    eps_advanced = step_eps(eps, omega_el, T_s, 0.5)
    u_abc = dq2abc(u_dq, eps_advanced)
    # clip in abc coordinates
    u_abc = jnp.clip(u_abc, -u_dc / 2.0, u_dc / 2.0)
    u_dq = abc2dq(u_abc, eps)
    return u_dq


def lut_interpolate(grid_x, grid_y, values, i_d, i_q):
    ix = jnp.searchsorted(grid_x, i_d) - 1
    iy = jnp.searchsorted(grid_y, i_q) - 1
    ix = jnp.clip(ix, 0, grid_x.shape[0] - 2)
    iy = jnp.clip(iy, 0, grid_y.shape[0] - 2)
    tx = (i_d - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix])
    ty = (i_q - grid_y[iy]) / (grid_y[iy + 1] - grid_y[iy])
    return (
        values[ix, iy] * (1 - tx) * (1 - ty)
        + values[ix + 1, iy] * tx * (1 - ty)
        + values[ix, iy + 1] * (1 - tx) * ty
        + values[ix + 1, iy + 1] * tx * ty
    )


class PMSM(eqx.Module):
    T_s: float = eqx.field(static=True)
    _solver: diffrax.AbstractSolver = eqx.field(static=True)
    env_properties: eqx.Module
    action_dim: int = eqx.field(static=True)
    physical_state_dim: int = eqx.field(static=True)
    LUT_grids: dict
    LUT_values: dict

    def __init__(
        self,
        saturated=False,
        motor_variant: MotorVariant = MotorVariant.DEFAULT,
        static_params: dict = None,
        solver=diffrax.Euler(),
        T_s: float = 1e-4,
    ):
        """
        Args:
            saturated (bool): Permanent magnet flux linkages and inductances are taken from LUT_motor_name
                specific LUTs. Default: False
            motor_variant (MotorVariant): Sets physical_normalizations, action_normalizations, soft_constraints
                and static_params to default values for the passed motor variant and stores associated LUTs for
                the possible saturated case. Needed if saturated==True.
            soft_constraints (Callable): Function that returns soft constraints values for state and/or action.
            static_params (dict): Parameters of environment which do not change during simulation.
                p (int): Pole pair number. Default: 3
                r_s (float): Stator resistance. Default: 15e-3
                l_d (float): Inductance in direct axes if motor not set to saturated. Default: 0.37e-3
                l_q (float): Inductance in quadrature axes if motor not set to saturated. Default: 65.6e-3,
                psi_p (float): Permanent magnet flux linkage if motor not set to saturated. Default: 122e-3,
                u_dc (float): dc-link voltage
                i_lim (float): voltage limit
            solver (diffrax.solver): Solver used to compute state for next step.
            T_s (float): Duration of one control/simulation step in seconds. Default: 1e-4.
        """
        if motor_variant != MotorVariant.DEFAULT:
            motor_params = motor_variant.get_params()
            default_static_params = motor_params.static_params.__dict__
            if saturated:
                default_static_params["l_d"] = jnp.nan
                default_static_params["l_q"] = jnp.nan
                default_static_params["psi_p"] = jnp.nan
                self.LUT_grids = motor_params.lut_grids
                self.LUT_values = motor_params.lut_values
            else:
                self.LUT_grids, self.LUT_values = self.generate_dummy_luts()

        else:
            if saturated:
                raise ValueError(
                    f"MotorVariant '{motor_variant.value}' is not allowed for saturated LUTs. "
                    "Use a specific motor variant. DEFAULT is only valid for saturated=False."
                )

            motor_params = motor_variant.get_params()
            default_static_params = motor_params.static_params.__dict__
            self.LUT_grids, self.LUT_values = self.generate_dummy_luts()

        if not static_params:
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
        """Dataclass containing the physical parameters of the environment."""

        p: jax.Array = eqx.field(converter=jnp.asarray)
        r_s: jax.Array = eqx.field(converter=jnp.asarray)
        l_d: jax.Array = eqx.field(converter=jnp.asarray)
        l_q: jax.Array = eqx.field(converter=jnp.asarray)
        psi_p: jax.Array = eqx.field(converter=jnp.asarray)
        u_dc: jax.Array = eqx.field(converter=jnp.asarray)
        i_lim: jax.Array = eqx.field(converter=jnp.asarray)

    class PhysicalState(eqx.Module):
        """Dataclass containing the physical state of the environment."""

        epsilon: jax.Array
        i_d: jax.Array
        i_q: jax.Array
        torque: jax.Array
        omega_el: jax.Array

    class Additions(eqx.Module):
        """Dataclass containing additional information for simulation."""

        solver_state: tuple
        active_solver_state: bool

    class Action(eqx.Module):
        """Dataclass containing the action, that can be applied to the environment."""

        u_d: jax.Array
        u_q: jax.Array

    class EnvProperties(eqx.Module):
        """Dataclass used for simulation which contains environment specific dataclasses."""

        saturated: bool = eqx.field(static=True)
        static_params: eqx.Module

    def generate_dummy_luts(self):
        saturated_quants = ["L_dd", "L_dq", "L_qd", "L_qq", "Psi_d", "Psi_q"]
        x_base = jnp.array([0.0, 1.0])
        y_base = jnp.array([0.0, 1.0])
        grids = {q: (x_base, y_base) for q in saturated_quants}
        values = {q: jnp.full((2, 2), jnp.nan) for q in saturated_quants}
        return grids, values

    def currents_to_torque(self, i_d, i_q):
        env_properties = self.env_properties
        torque = (
            1.5
            * env_properties.static_params.p
            * (
                env_properties.static_params.psi_p
                + (env_properties.static_params.l_d - env_properties.static_params.l_q) * i_d
            )
            * i_q
        )
        return torque

    def currents_to_torque_saturated(self, i_d, i_q):
        Psi_d = lut_interpolate(*self.LUT_grids["Psi_d"], self.LUT_values["Psi_d"], i_d, i_q)
        Psi_q = lut_interpolate(*self.LUT_grids["Psi_q"], self.LUT_values["Psi_q"], i_d, i_q)
        return 3 / 2 * self.env_properties.static_params.p * (Psi_d * i_q - Psi_q * i_d)

    def init_state(self):
        """Returns default initial state for all batches."""
        env_properties = self.env_properties
        phys = self.PhysicalState(
            epsilon=jnp.array(0.0),
            i_d=jnp.array(0.0),
            i_q=jnp.array(0.0),
            torque=jnp.array(0.0),
            omega_el=jnp.array(0.0),
        )
        rng = jnp.array(jnp.nan)

        def voltage(t):
            return jnp.array([0, 0])

        args = (env_properties.static_params, phys.omega_el)
        if env_properties.saturated:
            vector_field = partial(self.nonlinear_ode, action=voltage)
        else:
            vector_field = partial(self.linear_ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.T_s
        y0 = tuple([phys.i_d, phys.i_q, phys.epsilon])

        solver_state = self._solver.init(term, t0, t1, y0, args)
        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)

        return self.State(physical_state=phys, prng_key=rng, additions=additions)

    def nonlinear_ode(self, t, y, args, action):
        i_d, i_q, eps = y
        static_params, omega_el = args
        u_dq = action(t)
        J_k = jnp.array([[0, -1], [1, 0]])
        i_dq = jnp.array([i_d, i_q])
        p_d = {q: lut_interpolate(*self.LUT_grids[q], self.LUT_values[q], i_d, i_q) for q in self.LUT_grids}
        L_diff = jnp.column_stack([p_d[q] for q in ["L_dd", "L_dq", "L_qd", "L_qq"]]).reshape(2, 2)
        L_diff_inv = jnp.linalg.inv(L_diff)
        psi_dq = jnp.column_stack([p_d[psi] for psi in ["Psi_d", "Psi_q"]]).reshape(-1)
        di_dq_1 = jnp.einsum(
            "ij,j->i",
            (-L_diff_inv * static_params.r_s),
            i_dq,
        )
        di_dq_2 = jnp.einsum("ik,k->i", L_diff_inv, u_dq)
        di_dq_3 = jnp.einsum("ij,jk,k->i", -L_diff_inv, J_k, psi_dq) * omega_el
        i_dq_diff = di_dq_1 + di_dq_2 + di_dq_3
        eps_diff = omega_el
        d_y = i_dq_diff[0], i_dq_diff[1], eps_diff
        return d_y

    def linear_ode(self, t, y, args, action):
        i_d, i_q, eps = y
        params, omega_el = args
        u_dq = action(t)
        u_d = u_dq[0]
        u_q = u_dq[1]
        l_d = params.l_d
        l_q = params.l_q
        psi_p = params.psi_p
        r_s = params.r_s
        i_d_diff = (u_d + omega_el * l_q * i_q - r_s * i_d) / l_d
        i_q_diff = (u_q - omega_el * (l_d * i_d + psi_p) - r_s * i_q) / l_q
        eps_diff = omega_el
        d_y = i_d_diff, i_q_diff, eps_diff
        return d_y

    @eqx.filter_jit
    def _ode_solver_step(self, state, u_dq):
        """Computes state by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            u_dq: The action to apply to the environment.

        Returns:
            state: The computed state after the one step simulation.
        """
        properties = self.env_properties
        system_state = state.physical_state
        omega_el = system_state.omega_el
        i_d = system_state.i_d
        i_q = system_state.i_q
        eps = system_state.epsilon

        def voltage(t):
            return u_dq

        args = (properties.static_params, omega_el)
        if properties.saturated:
            vector_field = partial(self.nonlinear_ode, action=voltage)
        else:
            vector_field = partial(self.linear_ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.T_s
        y0 = tuple([i_d, i_q, eps])

        def false_fn(_):
            return self.Additions(solver_state=self._solver.init(term, t0, t1, y0, args), active_solver_state=True)

        def true_fn(_):
            return state.additions

        additions = jax.lax.cond(state.additions.active_solver_state, false_fn, true_fn, operand=None)

        y, _, _, solver_state_k1, _ = self._solver.step(term, t0, t1, y0, args, additions.solver_state, made_jump=False)

        i_d_k1 = y[0]
        i_q_k1 = y[1]
        eps_k1 = y[2]

        eps_k1 = ((eps_k1 + jnp.pi) % (2 * jnp.pi)) - jnp.pi

        if properties.saturated:
            torque = jnp.array([self.currents_to_torque_saturated(i_d=i_d_k1, i_q=i_q_k1)])[0]
        else:
            torque = jnp.array([self.currents_to_torque(i_d_k1, i_q_k1)])[0]

        new_physical_state = eqx.tree_at(
            lambda s: (s.epsilon, s.i_d, s.i_q, s.torque), system_state, (eps_k1, i_d_k1, i_q_k1, torque)
        )
        new_additions = self.Additions(solver_state=solver_state_k1, active_solver_state=True)
        new_state = eqx.tree_at(lambda s: (s.physical_state, s.additions), state, (new_physical_state, new_additions))
        return new_state

    def apply_voltage_constraint(self, u_dq, system_state):
        """Denormalizes the u_dq and clips it with respect to the hexagon."""
        env_properties = self.env_properties

        # normalize to u_dc/2 for hexagon constraints
        u_dq_norm = u_dq * (1 / (env_properties.static_params.u_dc / 2))
        advanced_angle = step_eps(
            system_state.physical_state.epsilon,
            omega_el=system_state.physical_state.omega_el,
            T_s=self.T_s,
            T_s_scale=1.5,
        )
        u_albet_norm = dq2albet(
            u_dq_norm,
            advanced_angle,
        )
        u_albet_norm_clip = apply_hex_constraint(u_albet_norm)
        u_dq_norm_clip = albet2dq(
            u_albet_norm_clip,
            advanced_angle,
        )
        # denormalize from u_dc/2
        u_dq = u_dq_norm_clip[0] * (env_properties.static_params.u_dc / 2)
        return u_dq

    @eqx.filter_jit
    def step(self, state, action):
        """Computes state by simulating one step.

        Args:
            system_state: The state from which to calculate state for the next step.
            action: The action to apply to the environment.
        Returns:
            state: The computed state after the one step simulation.
        """
        u_dq = self.apply_voltage_constraint(action, state)

        next_state = self._ode_solver_step(state, u_dq)
        observation = self.generate_observation(next_state)
        return observation, next_state

    @property
    def action_description(self):
        return ["u_d", "u_q"]

    @property
    def obs_description(self):
        _obs_description = [
            "i_d",
            "i_q",
            "cos_eps",
            "sin_eps",
            "omega_el",
            "torque",
        ]
        return np.hstack(
            [
                np.array(_obs_description),
                np.array([name + "_ref" for name in self.control_state]),
            ]
        )

    def generate_observation(self, state):
        """Returns observation for one batch."""
        eps = state.physical_state.epsilon
        cos_eps = jnp.cos(eps)
        sin_eps = jnp.sin(eps)
        state_phys = state.physical_state
        obs = jnp.hstack(
            (
                state_phys.i_d,
                state_phys.i_q,
                state_phys.omega_el,
                state_phys.torque,
                cos_eps,
                sin_eps,
            )
        )
        return obs

    @eqx.filter_jit
    def generate_state_from_observation(self, obs, key=None):
        """Generates state from observation for one batch."""
        env_properties = self.env_properties
        if key is not None:
            subkey = key
        else:
            subkey = jnp.nan
        phys = self.PhysicalState(
            epsilon=jnp.arctan2(obs[5], obs[4]) / jnp.pi,
            i_d=obs[0],
            i_q=obs[1],
            torque=obs[3],
            omega_el=obs[2],
        )

        def voltage(t):
            return jnp.array([0, 0])

        args = (env_properties.static_params, phys.omega_el)
        if env_properties.saturated:
            vector_field = partial(self.nonlinear_ode, action=voltage)
        else:
            vector_field = partial(self.linear_ode, action=voltage)

        term = diffrax.ODETerm(vector_field)
        t0 = 0
        t1 = self.T_s
        y0 = tuple([phys.i_d, phys.i_q, phys.epsilon])

        solver_state = self._solver.init(term, t0, t1, y0, args)

        dummy_solver_state = jax.tree.map(
            lambda x: jnp.full_like(x, jnp.nan) if jnp.issubdtype(x.dtype, jnp.floating) else x, solver_state
        )

        additions = self.Additions(solver_state=dummy_solver_state, active_solver_state=False)

        for i, name in enumerate(self.control_state):
            new_ref = eqx.tree_at(lambda r: getattr(r, name), new_ref, obs[8 + i])
        state = self.State(physical_state=phys, prng_key=subkey, additions=additions)
        return state

    def set_speed(self, n, state):
        omega_el = jnp.array(self.env_properties.static_params.p * n * 2 * jnp.pi / 60)
        new_state = eqx.tree_at(lambda r: r.physical_state.omega_el, state, omega_el)

        obs = self.generate_observation(new_state)
        return obs, new_state

    def reset(
        self,
    ):
        """
        Resets environment to default, random or passed initial state.

        Args:
            rng (optional): Random key for random initialization.
            initial_state (optional): The initial_state to which the environment will be reset.

        Returns:
            obs: Observation of initial state.
            state: The initial state.
        """

        state = self.init_state()
        obs = self.generate_observation(state)

        return obs, state
