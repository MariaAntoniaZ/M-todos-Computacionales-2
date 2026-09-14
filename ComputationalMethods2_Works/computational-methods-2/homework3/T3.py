import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import pandas as pd
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors



drag = pd.read_csv("drag.csv")

depth = drag.iloc[:,0].values
beta_vals = drag.iloc[:,1].values

betas = interp1d(depth, beta_vals, fill_value="extrapolate")

m = 1.0
n = 6.857

def sistema(t, Y):
    x, y, vx, vy = Y
    
    v = np.sqrt(vx**2 + vy**2)
    beta_y = betas(-y)
    
    ax = - beta_y * (v**2.31) * (vx / v)
    ay = n - beta_y * (v**2.31) * (vy / v)
    
    return np.array([vx, vy, ax, ay])

def evento(t, Y):
    return Y[1]

evento.terminal = 2

v_values = np.arange(1, 11, 0.1)

v0 = 10
angles = np.arange(-80, -11, 1)
theta = np.deg2rad(-40)
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)
sol = solve_ivp(
    sistema,
    t_span=(0, 5),
    y0=[0, 0, vx0, vy0],
    max_step=0.01,
    events=evento
)
x_final = sol.y_events[0][0][0]
x_final

angles = np.arange(-80, -11, 1)
theta_optimal = []

for v0 in tqdm(v_values):

    def neg_range(theta_deg):

        theta = np.deg2rad(theta_deg)

        vx0 = v0 * np.cos(theta)
        vy0 = v0 * np.sin(theta)

        sol = solve_ivp(
            sistema,
            t_span=(0, 50),
            y0=[0, 0.0, vx0, vy0],
            events=evento
        )

        x_min = -sol.y[0][-1]
        return x_min

    result = minimize_scalar(
        neg_range,
        bounds=(-80, -10),
        method='bounded'
    )
    theta_optimal.append(result.x)

plt.figure(figsize=(20,5))
plt.plot(v_values, theta_optimal)
plt.xlabel("Velocidad inicial [m/s]")
plt.ylabel("Optimal angle (degrees)")
plt.title("Alcance máximo bajo el agua con flotación g = 6.875")
plt.savefig("1.angle.pdf")
plt.close()





G = 1
m = 1.7
tmax = 150
dt = 0.01
N = int(tmax/dt)
ts = np.arange(0,tmax,dt)

def a(r1,r2):
    r = r1 - r2
    dist = np.linalg.norm(r)
    a1 = -G * m * r / dist**3
    a2 = -a1
    return a1, a2

def E(y):
    r1 = y[0:2]
    r2 = y[2:4]
    v1 = y[4:6]
    v2 = y[6:8]

    r = np.linalg.norm(r1-r2)
    k = 0.5*m*(np.dot(v1,v1) + np.dot(v2,v2))
    u = -G*m*m/r

    return k + u

def L(y):
    r1 = y[0:2]
    r2 = y[2:4]
    v1 = y[4:6]
    v2 = y[6:8]

    L = m*(np.cross(r1, v1) + np.cross(r2, v2))
    return L

def f(t, y):
    r1 = y[0:2]
    r2 = y[2:4]
    v1 = y[4:6]
    v2 = y[6:8]

    r = r1 - r2
    dist = np.linalg.norm(r)

    a1 = -G*m * r / dist**3
    a2 =  G*m * r / dist**3

    dydt = np.zeros_like(y)
    dydt[0:2] = v1
    dydt[2:4] = v2
    dydt[4:6] = a1
    dydt[6:8] = a2

    return dydt

def rk38_step(t, y, dt):

    k1 = f(t, y)
    k2 = f(t + dt/3,
           y + dt*(1/3)*k1)
    k3 = f(t + 2*dt/3,
           y + dt*(-1/3*k1 + k2))
    k4 = f(t + dt,
           y + dt*(k1 - k2 + k3))

    y_next = y + dt*(1/8*k1 + 3/8*k2 + 3/8*k3 + 1/8*k4)

    return y_next

def verlet_step(t, y, dt):

    r1 = y[0:2]
    r2 = y[2:4]
    v1 = y[4:6]
    v2 = y[6:8]

    a1, a2 = a(r1, r2)

    r1_new = r1 + v1*dt + 0.5*a1*dt**2
    r2_new = r2 + v2*dt + 0.5*a2*dt**2

    a1_new, a2_new = a(r1_new, r2_new)

    v1_new = v1 + 0.5*(a1 + a1_new)*dt
    v2_new = v2 + 0.5*(a2 + a2_new)*dt

    y_next = np.zeros_like(y)
    y_next[0:2] = r1_new
    y_next[2:4] = r2_new
    y_next[4:6] = v1_new
    y_next[6:8] = v2_new

    return y_next

y0 = np.array([0.,0.,
               1.,1.,
               0.,0.5,
               0.,-0.5])


y = y0.copy()
t = 0

r1_rk = []
r2_rk = []
E_rk = []
L_rk = []

for i in range(N):
    r1_rk.append(y[0:2].copy())
    r2_rk.append(y[2:4].copy())
    E_rk.append(E(y))
    L_rk.append(L(y))
    y = rk38_step(t, y, dt)
    t += dt


y = y0.copy()
t = 0

r1_vv = []
r2_vv = []
E_vv = []
L_vv = []

for i in range(N):
    r1_vv.append(y[0:2].copy())
    r2_vv.append(y[2:4].copy())
    E_vv.append(E(y))
    L_vv.append(L(y))
    y = verlet_step(t, y, dt)
    t += dt

r1_rk = np.array(r1_rk)
r2_rk = np.array(r2_rk)
r1_vv = np.array(r1_vv)
r2_vv = np.array(r2_vv)

r1_vv

t_values = np.arange(0, tmax, dt)

fig, ax = plt.subplots(3, 2, figsize=(12, 12))

ax[0,0].plot(r1_vv[:,0], r1_vv[:,1])
ax[0,0].plot(r2_vv[:,0], r2_vv[:,1])
ax[0,0].set_title("Velocity-Verlet Orbit")

ax[0,1].plot(r1_rk[:,0], r1_rk[:,1])
ax[0,1].plot(r2_rk[:,0], r2_rk[:,1])
ax[0,1].set_title("Runge-Kutta 3/8 Orbit")

ax[1,0].plot(t_values, E_vv)
ax[1,0].set_title("Velocity-Verlet Energy")

ax[1,1].plot(t_values, E_rk)
ax[1,1].set_title("Runge-Kutta 3/8 Energy")

ax[2,0].plot(t_values, L_vv)
ax[2,0].set_title("Velocity-Verlet Angular Momentum")

ax[2,1].plot(t_values, L_rk)
ax[2,1].set_title("Runge-Kutta 3/8 Angular Momentum")

plt.tight_layout()
fig.savefig("2.pdf")
plt.close()








a = 0.7
b = 0.8
R = 1
tmax = 1700


def fitzhugh_nagumo(t, y, tau, Iext):
    v, w = y
    dv = v - (v**3)/3 - w + R*Iext
    dw = (v + a - b*w) / tau
    return [dv, dw]


tau_vals = np.logspace(-0.3, 2.1)
I_vals = np.linspace(0.2, 4, 100)

freq_v = np.zeros((len(I_vals), len(tau_vals)))
freq_w = np.zeros((len(I_vals), len(tau_vals)))


t_span = (0, tmax)
t_eval = np.linspace(0, tmax, 1000)

for i, Iext in tqdm(list(enumerate(I_vals))):
    for j, tau in enumerate(tau_vals):

        sol = solve_ivp(fitzhugh_nagumo, t_span, [0,0],
                        args=(tau, Iext),
                        t_eval=t_eval,
                        rtol=1e-6)

        v = sol.y[0]
        w = sol.y[1]
        t = sol.t

        mask = t > 50
        t = t[mask]
        v = v[mask]
        w = w[mask]

        peaks_v, _ = find_peaks(v, prominence=0.1)
        if len(peaks_v) > 2:
            periods = np.diff(t[peaks_v])
            freq_v[i,j] = 1/np.mean(periods)
        else:
            freq_v[i,j] = 0

        peaks_w, _ = find_peaks(w, prominence=0.05)
        if len(peaks_w) > 2:
            periods = np.diff(t[peaks_w])
            freq_w[i,j] = 1/np.mean(periods)
        else:
            freq_w[i,j] = 0


T, I = np.meshgrid(tau_vals, I_vals)

fig, ax = plt.subplots(1,2, figsize=(12,5))

c1 = ax[0].contourf(T, I, freq_v, levels=100, cmap='viridis', norm = 'log')
ax[0].set_title('Frecuencia de v')
ax[0].set_xlabel(r'$\tau$')
ax[0].set_ylabel(r'$I_{ext}$')
ax[0].set_xscale('log')

c2 = ax[1].contourf(T, I, freq_w, levels=100, cmap='viridis', norm = 'log')
ax[1].set_title('Frecuencia de w')
ax[1].set_xlabel(r'$\tau$')
ax[1].set_ylabel(r'$I_{ext}$')
ax[1].set_xscale('log')

plt.tight_layout()
plt.savefig("3.C.pdf")
plt.close()










def sistema(t, y, alpha):
    theta, r, P_theta, P_r = y
    
    dtheta = P_theta / (r + 1)**2
    dr = P_r
    dP_theta = -alpha**2 * (r + 1) * np.sin(theta)
    dP_r = alpha**2 * np.cos(theta) - r + P_theta**2 / (1 + r)**3
    
    return [dtheta, dr, dP_theta, dP_r]


def evento_theta_cero(t, y, alpha):
    return y[0]

evento_theta_cero.terminal = False
evento_theta_cero.direction = 0

y0 = [np.pi/2, 0, 0, 0]

alphas = np.linspace(0.7, 1.5, 25)
t_span = (0, 10000)

plt.style.use("default")

fig, ax = plt.subplots(figsize=(8,6))

norm = mcolors.Normalize(vmin=min(alphas), vmax=max(alphas))
cmap = cm.magma  

for alpha in alphas:
    
    sol = solve_ivp(
        sistema,
        t_span,
        y0,
        args=(alpha,),
        events=evento_theta_cero,
        max_step=0.1,
        rtol=1e-8,
        atol=1e-10
    )
    
    eventos = sol.y_events[0]
    
    if len(eventos) > 0:
        r_vals = eventos[:,1]
        Pr_vals = eventos[:,3]
        
        ax.scatter(
            r_vals,
            Pr_vals,
            color=cmap(norm(alpha)),  
            s=3,                      
            alpha=0.6                 
        )

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$\alpha$", fontsize=12)

ax.set_xlabel("r", fontsize=12)
ax.set_ylabel(r"$P_r$", fontsize=12)
ax.set_title("Secciones de Poincaré", fontsize=14)

ax.grid(False)

plt.tight_layout()
plt.savefig("4.pdf", dpi=300)
plt.close(fig)








θ_0 = 1 + 0j
θd_0 = 0 + 0j

def profile(xi, y, n):
    θ = y[0]
    ω = y[1]

    dθ_dxi = ω

    if xi == 0:
        dω_dxi = -1/3
    else:
        dω_dxi = -θ**n - (2/xi)*ω

    return [dθ_dxi, dω_dxi]



def evento(xi, y, n):
    return y[0]
evento.terminal = True

Rs = []
Ms = []
rel_ρs = []

ns = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0]

for ni in ns:
    sol = solve_ivp(
        fun=lambda xi, y: profile(xi, y, n=ni),
        t_span=(0, 200),
        y0=[θ_0, θd_0],
        events = lambda xi, y: evento(xi, y, n=ni),
        atol=1e-10,
        rtol=1e-10,
        max_step = 10e-2
    )

    θ, θd = sol.y_events[0][0]
    ξ = sol.t_events[0][0]
    M = np.real(-(ξ**2) * θd)
    rel_ρ = np.real(-1/3 * (ξ/θd)) 

    Rs.append(round(ξ,5))
    Ms.append(round(M,5))
    rel_ρs.append(round(rel_ρ,5))

df = pd.DataFrame({
    "n": ns,
    "R": Rs,
    "M": Ms,
    "ρ₀/<ρ>": rel_ρs
})

df.to_csv("7.csv", index=False)

