import numpy as np, jax, jax.numpy as jnp, matplotlib.pyplot as plt
from typing import List, Dict
from functools import partial
import jaxopt, cvxpy as cp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML



class ControlAffineSystem():

    def __init__(self, **kwargs):
            
        self.dim_x = kwargs.get('dim_x', 3) # by default a 3d state space (x,y, theta)
        self.dim_u = kwargs.get('dim_u', 1) # by default a 1d control input
        self.T = kwargs.get('T', 20) # time horizon
        self.dt = kwargs.get('dt', 0.08)

        self.cbf_rate = kwargs.get('cbf_rate', 0.5)
        self.clf_rate = kwargs.get('clf_rate', 0.5)
        self.slack_weight = kwargs.get('slack_weight', 1.0)

        self.fn_f:callable = kwargs.get('fn_f', None)  # open loop dynamics
        self.fn_g:callable = kwargs.get('fn_g', None)  # control dynamics
        self.fn_clf:callable = kwargs.get('fn_clf', None)
        self.fn_cbf:callable = kwargs.get('fn_cbf', None)
        self.fn_clf_grad:callable = kwargs.get('fn_clf_grad', None)
        self.fn_cbf_grad:callable = kwargs.get('fn_cbf_grad', None)

    
    def fn_dynamics_ode(self, state, control):
        if control.size == 1:
            state_dot = self.fn_f(state) + self.fn_g(state) * control
        else:
            state_dot = self.fn_f(state) + self.fn_g(state) @ control
        return state_dot


    def solve_qp(self, state, control):
        raise NotImplementedError("solve_clf_cbf_pq method is not implemented")


    def plot_trajectory(self, trajectory):
        raise NotImplementedError("plot_trajectory method is not implemented")

    
    def animate_trajectory(self, trajectory):
        raise NotImplementedError("animate_trajectory method is not implemented")


class DubinsCar(ControlAffineSystem):


    def __init__(self, **kwargs):
        
        self.speed = kwargs.get('speed', 1.0)
        self.max_turn_rate = kwargs.get('max_turn_rate', 3.0)
        self.min_separation = kwargs.get('obs_r', 2.5)
        self.obs_xy = kwargs.get('obs_xy', [5,4])
        self.goal_xy = kwargs.get('goal_xy', [12,0])
        self.state_init = kwargs.get('state_init', jnp.array([0.,5.,0]))
        self.control_init = kwargs.get('control_init', jnp.array([0.]))
        
        @jax.jit
        def _fn_f(state):
            x, y, theta = state
            return jnp.array([self.speed*jnp.cos(theta), 
                              self.speed*jnp.sin(theta), 
                              0])

        @jax.jit
        def _fn_g(state):
            return jnp.array([0,0,1])

        @jax.jit
        def _fn_cbf(state):
            x, y, theta = state
            fn_dist_sq = lambda _state: jnp.dot(_state[:2] - jnp.array(self.obs_xy),
                                                _state[:2] - jnp.array(self.obs_xy)) - self.min_separation ** 2
            dist_sq = fn_dist_sq(state)
            grad_dist_sq = jnp.dot(jax.grad(fn_dist_sq)(state), _fn_f(state))
            
            # augment cbf with grad w.r.t. theta: need this to introduce theta in CBF, 
            # otherwise we lose the constraint on control input theta_dot in the QP
            return dist_sq + grad_dist_sq

        @jax.jit
        def _fn_clf(state):
            x, y, theta = state
            dist = jnp.cos(theta) * (y - self.goal_xy[1]) \
                 - jnp.sin(theta) * (x - self.goal_xy[0])
            return dist**2

        super().__init__(**kwargs, 
                         fn_f=_fn_f, fn_g=_fn_g, fn_cbf=_fn_cbf, fn_clf=_fn_clf,
                         fn_cbf_grad=jax.grad(_fn_cbf), fn_clf_grad=jax.grad(_fn_clf))


    @partial(jax.jit, static_argnums=(0,))
    def solve_qp(self, state, control):

        B = self.fn_cbf(state)
        V = self.fn_clf(state)
        fx = self.fn_f(state)
        gx = self.fn_g(state)
        cbf_grad = self.fn_cbf_grad(state)
        clf_grad = self.fn_clf_grad(state)
        LfB = jnp.dot(cbf_grad, fx)
        LgB = jnp.dot(cbf_grad, gx)
        LfV = jnp.dot(clf_grad, fx)
        LgV = jnp.dot(clf_grad, gx)
        
        Q = jnp.array([[1.0, 0.0],  # objective: min 0.5 x^T Q x + c^T x
                       [0.0, self.slack_weight]])
        c = jnp.zeros(len(control)+1)
        
        G = jnp.vstack([            # constraints: Gx<=h
            jnp.array([LgV, -1.0]),
            jnp.array([-LgB, 0.0]),
            jnp.hstack([jnp.eye(len(control)), jnp.zeros((len(control), 1))]),
            jnp.hstack([-jnp.eye(len(control)), jnp.zeros((len(control), 1))]),
        ])
        h = jnp.array([
            -LfV - self.clf_rate * V,  # CLF constraint
            LfB + self.cbf_rate * B,   # CBF constraint
            self.max_turn_rate,        # Control max bound
            self.max_turn_rate         # Control min bound
        ])

        try:
            prob = jaxopt.OSQP()
            sol = prob.run(params_obj=(Q, c), params_ineq=(G, h)).params
            control = sol.primal[:-1]
            stat_infeasible = False
        except Exception as e:
            print(e)
            stat_infeasible = True
        
        return control, stat_infeasible
    
    
    def plot_trajectory(self, trajectory):
        fig = plt.figure(figsize=(20,6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        ax1.set_title('State Trajectory'); ax1.axis('equal')
        ax2.set_title(f"Control signals and solver status")
        ax3.set_title(f"CBF & CLF value")
        
        ts = np.arange(0, self.T, self.dt)
        
        ax1.add_patch(plt.Circle(self.obs_xy, self.min_separation, color='gray', alpha=0.3, label='obstacle'))
        ax1.add_patch(plt.Circle(self.goal_xy, 0.1, color='red', alpha=0.3, label='goal'))
        points=ax1.scatter(trajectory['state'][:,0], trajectory['state'][:,1], c=ts, cmap='viridis')
        cbar = plt.colorbar(points, ax=ax1)
        cbar.set_label('Time (s)')
        
        stat_infeasible = trajectory['stat_infeasible']
        ax2.plot(ts, trajectory['control'], label=r'$\omega$')
        ax2.axhline(self.max_turn_rate, c='gray', ls='--', label='max turn rate', alpha=0.1)
        ax2.axhline(-self.max_turn_rate, c='gray', ls='--', alpha=0.1)
        ax2.plot(ts, stat_infeasible*10, alpha=0.2, label='qp failure?', c='red', ls=':')
        
        cbf_val = [self.fn_cbf(trajectory['state'][i]) for i in range(len(ts))]
        clf_val = [self.fn_clf(trajectory['state'][i]) for i in range(len(ts))]
        ax3.plot(ts, cbf_val, label='cbf')
        ax3.plot(ts, clf_val, label='clf')
        ax3.axhline(0, c='gray', ls='--', alpha=0.4, label='CBF=0', color='red')
        ax1.legend(); ax2.legend(); ax3.legend()
        
        
class TwoDubinsCar(ControlAffineSystem):


    def __init__(self, **kwargs):
        
        self.speed = kwargs.get('speed', 1.0)
        self.max_turn_rate = kwargs.get('max_turn_rate', 3.0)
        self.min_separation = kwargs.get('obs_r', 2.5)
        self.state_init = kwargs.get('state_init', jnp.array([-5.,0,0.1,5,0,-jnp.pi])) # x1, y1, th1, x2, y2, th2
        self.control_init = kwargs.get('control_init', jnp.array([0.])) # th1_dot
        
        @jax.jit
        def _fn_f(state):
            x1, y1, th1, x2, y2, th2 = state
            return jnp.array([self.speed*jnp.cos(th1), self.speed*jnp.sin(th1), 0,
                              self.speed*jnp.cos(th2), self.speed*jnp.sin(th2), 0])

        @jax.jit
        def _fn_g(state):
            return jnp.array([0,0,1,0,0,0])

        @jax.jit
        def _fn_cbf(state):
            x1, y1, th1, x2, y2, th2 = state
            fn_dist_sq = lambda _state: jnp.dot(_state[0:2] - _state[3:5],
                                                _state[0:2] - _state[3:5]) - self.min_separation ** 2
            dist_sq = fn_dist_sq(state)
            grad_dist_sq = jnp.dot(jax.grad(fn_dist_sq)(state), _fn_f(state))

            return dist_sq + grad_dist_sq

        @jax.jit
        def _fn_clf(state):
            x1, y1, th1, x2, y2, th2 = state
            
            # dist = jnp.cos(th1) * (y1 - y2) - jnp.sin(th1) * (x1 - x2) # facilitates chasing
            dist = jnp.cos(th1) * jnp.sin(th2) - jnp.sin(th1) * jnp.cos(th2) + (y1-y2) # maintains course
            return dist**2


        super().__init__(**kwargs, 
                         fn_f=_fn_f, fn_g=_fn_g, fn_cbf=_fn_cbf, fn_clf=_fn_clf,
                         fn_cbf_grad=jax.grad(_fn_cbf), fn_clf_grad=jax.grad(_fn_clf))


    @partial(jax.jit, static_argnums=(0,))
    def solve_qp(self, state, control):

        B = self.fn_cbf(state)
        V = self.fn_clf(state)
        fx = self.fn_f(state)
        gx = self.fn_g(state)
        cbf_grad = self.fn_cbf_grad(state)
        clf_grad = self.fn_clf_grad(state)
        LfB = jnp.dot(cbf_grad, fx)
        LgB = jnp.dot(cbf_grad, gx)
        LfV = jnp.dot(clf_grad, fx)
        LgV = jnp.dot(clf_grad, gx)
        
        # objective: min 0.5 x^T Q x + c^T x
        Q = jnp.array([[1.0, 0.0],  
                       [0.0, self.slack_weight]])
        c = jnp.zeros(len(control)+1)
        
        # constraints: Gx<=h
        G = jnp.vstack([            
            jnp.array([LgV, -1.0]),
            jnp.array([-LgB, 0.0]),
            jnp.hstack([jnp.eye(len(control)), jnp.zeros((len(control), 1))]),
            jnp.hstack([-jnp.eye(len(control)), jnp.zeros((len(control), 1))]),
        ])
        h = jnp.array([
            -LfV - self.clf_rate * V,  # CLF constraint
            LfB + self.cbf_rate * B,   # CBF constraint
            self.max_turn_rate,        # Control max bound
            self.max_turn_rate         # Control min bound
        ])

        try:
            prob = jaxopt.OSQP()
            sol = prob.run(params_obj=(Q, c), params_ineq=(G, h)).params
            control = sol.primal[:-1]
            stat_infeasible = False
        except Exception as e:
            print(e)
            stat_infeasible = True
        
        return control, stat_infeasible
    
    
    def plot_trajectory(self, trajectory):
        fig = plt.figure(figsize=(20,6))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        ax1.set_title('State Trajectory'); ax1.axis('equal')
        ax2.set_title(f"Control signals and solver status")
        ax3.set_title(f"CBF & CLF value")
        
        ts = np.arange(0, self.T, self.dt)

        points1=ax1.scatter(trajectory['state'][:,0], trajectory['state'][:,1], c=ts, cmap='Blues')
        points2=ax1.scatter(trajectory['state'][:,3], trajectory['state'][:,4], c=ts, cmap='Reds')
        cbar1 = plt.colorbar(points1, ax=ax1)
        cbar2 = plt.colorbar(points2, ax=ax1)
        cbar1.set_label('Time (s): Car 1')
        cbar2.set_label('Time (s): Car 2')
        
        stat_infeasible = trajectory['stat_infeasible']
        ax2.plot(ts, trajectory['control'], label=r'$\omega$')
        ax2.axhline(self.max_turn_rate, c='gray', ls='--', label='max turn rate', alpha=0.1)
        ax2.axhline(-self.max_turn_rate, c='gray', ls='--', alpha=0.1)
        ax2.plot(ts, stat_infeasible*10, alpha=0.2, label='qp failure?', c='red', ls=':')

        clf_val = [self.fn_clf(trajectory['state'][i]) for i in range(len(ts))]
        cbf_val = [self.fn_cbf(trajectory['state'][i]) for i in range(len(ts))]
        ax3.plot(ts, clf_val, label='clf')
        ax3.plot(ts, np.log(cbf_val), label='ln(cbf)')
        ax3.axhline(0, c='gray', ls='--', alpha=0.4, label='CBF=0', color='red')
        ax1.legend(); ax2.legend(); ax3.legend()


    def animate_trajectory(self, trajectory):
        fig = plt.figure(figsize=(20,6))
        ax = fig.add_subplot(111)
        ax.set_title('State Trajectory'); ax.axis('equal')
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 15)
        ax.set_aspect('equal')
        
        ts = np.arange(0, self.T, self.dt)
        points1, = ax.plot([], [], 'bo')
        points2, = ax.plot([], [], 'ro')

        def init():
            points1.set_data([], [])
            points2.set_data([], [])
            return points1, points2

        def update(frame):
            points1.set_data(trajectory['state'][:frame, 0], trajectory['state'][:frame, 1])
            points2.set_data(trajectory['state'][:frame, 3], trajectory['state'][:frame, 4])
            return points1, points2

        ani = FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=True)
        return HTML(ani.to_jshtml(fps=50))
    
        