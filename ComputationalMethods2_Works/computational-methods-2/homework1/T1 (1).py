# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from IPython.display import clear_output
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d, PPoly
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline, make_smoothing_spline
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import norm
from scipy.signal import find_peaks
from numpy.linalg import norm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar

from scipy.integrate import quad

from scipy.signal import peak_widths








# nueva carga de datos

dataframes_Mo = []
for i in range(10, 50):
    df = pd.read_csv(f"Mo_unfiltered_10kV-50kV/Mo_{i}kV.dat", encoding="ISO8859", sep="\t", comment="#", names=["energy", "flux"])
    dataframes_Mo.append(df)


dataframes_W = []
for i in range(10, 50):
    df = pd.read_csv(f"W_unfiltered_10kV-50kV/W_{i}kV.dat", encoding="ISO8859", sep="\t", comment="#", names=["energy", "flux"])
    dataframes_W.append(df)


dataframes_Rh = []
for i in range(10, 50):
    df = pd.read_csv(f"Rh_unfiltered_10kV-50kV/Rh_{i}kV.dat",encoding="ISO8859", sep="\t", comment="#", names=["energy", "flux"])
    dataframes_Rh.append(df)






"""Molibdeno"""

def remove_peaks(df):
    E, F = df['energy'], df['flux']

    peaks, properties = find_peaks(
        F,
        prominence=0.05*np.max(F),
        distance=1
    )


    E_peaks = E[peaks]

    mask = np.ones_like(E, dtype=bool)

    width = 1

    for Ep in E_peaks:
        mask &= np.abs(E - Ep) > width

    spline = make_smoothing_spline(E[mask], F[mask], lam = 0.08)

    return spline

fluxes_refined_Mo = []
for df in dataframes_Mo:
    fluxes_refined_Mo.append(remove_peaks(df))


"""
fig, ax = plt.subplots()

line_refined, = ax.plot([], [], lw=2, label='Refined Flux')
line_original, = ax.plot([], [], lw=1, linestyle='--', label='Original Flux')

ax.legend()

def init():
    line_refined.set_data([], [])
    line_original.set_data([], [])
    return line_refined, line_original

def update(frame):
    line_refined.set_data(np.linspace(min(dataframes_Mo[frame]['energy']),max(dataframes_Mo[frame]['energy']),1000), fluxes_refined_Mo[frame](np.linspace(min(dataframes_Mo[frame]['energy']),max(dataframes_Mo[frame]['energy']),1000)))
    line_original.set_data(dataframes_Mo[frame]['energy'], dataframes_Mo[frame]['flux'])
    ax.relim()
    ax.autoscale_view()
    #ax.set_yscale('asinh')
    return line_refined, line_original

ani = FuncAnimation(
    fig,
    update,
    frames=range(len(dataframes_Mo)),
    init_func=init,
    interval=1000,
    blit=True,
    repeat=False
)

HTML(ani.to_jshtml())

"""

"""Tungsteno"""


fluxes_refined_W = []
for df in dataframes_W:
    fluxes_refined_W.append(remove_peaks(df))



"""
fig, ax = plt.subplots()

line_refined, = ax.plot([], [], lw=2, label='Refined Flux')
line_original, = ax.plot([], [], lw=1, linestyle='--', label='Original Flux')

ax.legend()

def init():
    line_refined.set_data([], [])
    line_original.set_data([], [])
    return line_refined, line_original

def update(frame):
    line_refined.set_data(np.linspace(min(dataframes_W[frame]['energy']),max(dataframes_W[frame]['energy']),1000), fluxes_refined_W[frame](np.linspace(min(dataframes_W[frame]['energy']),max(dataframes_W[frame]['energy']),1000)))
    line_original.set_data(dataframes_W[frame]['energy'], dataframes_W[frame]['flux'])
    ax.relim()
    ax.autoscale_view()
    ax.set_yscale('asinh')
    return line_refined, line_original

ani = FuncAnimation(
    fig,
    update,
    frames=range(len(dataframes_W)),
    init_func=init,
    interval=1000,
    blit=True,
    repeat=False
)

HTML(ani.to_jshtml())

"""

"""Rodio"""


fluxes_refined_Rh = []
for df in dataframes_Rh:
    fluxes_refined_Rh.append(remove_peaks(df))

"""
fig, ax = plt.subplots()

line_refined, = ax.plot([], [], lw=2, label='Refined Flux')
line_original, = ax.plot([], [], lw=1, linestyle='--', label='Original Flux')

ax.legend()

def init():
    line_refined.set_data([], [])
    line_original.set_data([], [])
    return line_refined, line_original

def update(frame):
    line_refined.set_data(np.linspace(min(dataframes_Rh[frame]['energy']),max(dataframes_Rh[frame]['energy']),1000), fluxes_refined_Rh[frame](np.linspace(min(dataframes_Rh[frame]['energy']),max(dataframes_Rh[frame]['energy']),1000)))
    line_original.set_data(dataframes_Rh[frame]['energy'], dataframes_Rh[frame]['flux'])
    ax.relim()
    ax.autoscale_view()
    ax.set_yscale('asinh')
    return line_refined, line_original

ani = FuncAnimation(
    fig,
    update,
    frames=range(len(dataframes_W)),
    init_func=init,
    interval=1000,
    blit=True,
    repeat=False
)

HTML(ani.to_jshtml())

"""

"""# Análisis barriga"""

#Optimize


max_fluxes_Mo, max_energies_Mo = [], []
for i in range(len(fluxes_refined_Mo)):
  lol = lambda E:-fluxes_refined_Mo[i](E)
  xd = minimize_scalar(lol, bounds = (0, 10 + i), method = 'bounded')
  max_x, max_y = xd.x, -xd.fun
  max_fluxes_Mo.append(max_y)
  max_energies_Mo.append(max_x)

max_fluxes_W, max_energies_W = [], []
for i in range(len(fluxes_refined_W)):
  lol = lambda E:-fluxes_refined_W[i](E)
  xd = minimize_scalar(lol, bounds = (0, 10 + i), method = 'bounded')
  max_x, max_y = xd.x, -xd.fun
  max_fluxes_W.append(max_y)
  max_energies_W.append(max_x)

max_fluxes_Rh, max_energies_Rh = [], []
for i in range(len(fluxes_refined_Rh)):
  lol = lambda E:-fluxes_refined_Rh[i](E)
  xd = minimize_scalar(lol, bounds = (0, 10 + i), method = 'bounded')
  max_x, max_y = xd.x, -xd.fun
  max_fluxes_Rh.append(max_y)
  max_energies_Rh.append(max_x)

#1. flujos máximos

energy = range(10,50)

"""

plt.plot(energy, max_fluxes_Mo, label = 'Mo')
plt.scatter(energy, max_fluxes_Mo, s = 5)

plt.plot(energy, max_fluxes_W, label = 'W')
plt.scatter(energy, max_fluxes_W, s = 5)

plt.plot(energy, max_fluxes_Rh, label = 'Rh')
plt.scatter(energy, max_fluxes_Rh, s = 5)


plt.legend(loc = 'best')
plt.yscale('log')
plt.xscale('log')
plt.grid(which = 'both')
plt.xlabel('log voltage (kV)')
plt.ylabel('log max Flux (a.u.)')
plt.style.use('dark_background')
"""

#2. E para Fmax

"""

plt.plot(energy, max_energies_Mo, label = 'Mo')
plt.scatter(energy, max_energies_Mo, s = 5)

plt.plot(energy, max_energies_W, label = 'W')
plt.scatter(energy, max_energies_W, s = 5)

plt.plot(energy, max_energies_Rh, label = 'Rh')
plt.scatter(energy, max_energies_Rh, s = 5)

plt.legend(loc = 'best')
plt.xlabel('Voltage (kV)')
plt.grid()
plt.ylabel('Energy at Max Flux (keV)')
"""


#3. curvatures
curvature_funcs_Mo = []
for i in range(len(fluxes_refined_Mo)):
  secderiv_bspline = fluxes_refined_Mo[i].derivative(nu=2)
  curvature_funcs_Mo.append(secderiv_bspline)

curvature_funcs_W = []
for i in range(len(fluxes_refined_W)):
  secderiv_bspline = fluxes_refined_W[i].derivative(nu=2)
  curvature_funcs_W.append(secderiv_bspline)

curvature_funcs_Rh = []
for i in range(len(fluxes_refined_Rh)):
  secderiv_bspline = fluxes_refined_Rh[i].derivative(nu=2)
  curvature_funcs_Rh.append(secderiv_bspline)

curvature_emax_Mo = []
for i in range(len(max_energies_Mo)):
  curvature_emax_Mo.append(curvature_funcs_Mo[i](max_energies_Mo[i]))

curvature_emax_W = []
for i in range(len(max_energies_W)):
  curvature_emax_W.append(curvature_funcs_W[i](max_energies_W[i]))

curvature_emax_Rh = []
for i in range(len(max_energies_Rh)):
  curvature_emax_Rh.append(curvature_funcs_Rh[i](max_energies_Rh[i]))




energy_linspaces_Mo = []
for df in dataframes_Mo:
  energy_linspace = np.linspace(min(df['energy']), max(df['energy']), 1000)
  energy_linspaces_Mo.append(energy_linspace)

energy_linspaces_W = []
for df in dataframes_W:
  energy_linspace = np.linspace(min(df['energy']), max(df['energy']), 1000)
  energy_linspaces_W.append(energy_linspace)

energy_linspaces_Rh = []
for df in dataframes_Rh:
  energy_linspace = np.linspace(min(df['energy']), max(df['energy']), 1000)
  energy_linspaces_Rh.append(energy_linspace)

#4. FWHM


def get_fwhm(F, E_linspace):
  peaks, _ = find_peaks(F, height = 2)
  if len(peaks) > 0:
    widths = peak_widths(F, peaks, rel_height=0.5)
    fwhm_samples = widths[0][0]
    delta_E = E_linspace[1] - E_linspace[0]
    return fwhm_samples * delta_E
  else:
    return np.nan

for i in range(len(dataframes_Mo)):
  f = fluxes_refined_Mo[i](energy_linspaces_Mo[i])
  peaks, _ = find_peaks(f, height = 2)
  fwhm = peak_widths(f, peaks, rel_height=0.5)



for i in range(len(dataframes_W)):
  f = fluxes_refined_W[i](energy_linspaces_W[i])
  peaks, _ = find_peaks(f, height = 2)
  fwhm = peak_widths(f, peaks, rel_height=0.5)


for i in range(len(dataframes_Rh)):
  f = fluxes_refined_Rh[i](energy_linspaces_Rh[i])
  peaks, _ = find_peaks(f, height = 2)
  fwhm = peak_widths(f, peaks, rel_height=0.5)


#limpiar el output, esta celda es para ver si esta bien

fwhms_Mo = []
for i in range(len(dataframes_Mo)):
  f = fluxes_refined_Mo[i](energy_linspaces_Mo[i])
  fwhm = get_fwhm(f, energy_linspaces_Mo[i])
  fwhms_Mo.append(fwhm)

fwhms_W = []
for i in range(len(dataframes_W)):
  f = fluxes_refined_W[i](energy_linspaces_W[i])
  fwhm = get_fwhm(f, energy_linspaces_W[i])
  fwhms_W.append(fwhm)

fwhms_Rh = []
for i in range(len(dataframes_Rh)):
  f = fluxes_refined_Rh[i](energy_linspaces_Rh[i])
  fwhm = get_fwhm(f, energy_linspaces_Rh[i])
  fwhms_Rh.append(fwhm)




#5. Skewness y kurtosis

def skewness_kurtosis(spline, Emin, Emax, N=5000):

    E = np.linspace(Emin, Emax, N)
    F = spline(E)

    F[F < 0] = 0

    B = np.trapz(F, E)

    E_mean = np.trapz(E * F, E) / B

    variance = np.trapz((E - E_mean)**2 * F, E) / B
    sigma = np.sqrt(variance)

    skew = np.trapz(((E - E_mean)/sigma)**3 * F, E) / B

    kurt = np.trapz(((E - E_mean)/sigma)**4 * F, E) / B

    return skew, kurt

skew_Mo, kurt_Mo = [], []

for i in range(len(fluxes_refined_Mo)):
  skew, kurt = skewness_kurtosis(
    fluxes_refined_Mo[i],
    Emin=energy_linspaces_Mo[i].min(),
    Emax=energy_linspaces_Mo[i].max()
  )
  skew_Mo.append(skew)
  kurt_Mo.append(kurt)



from scipy.integrate import quad
def skewness_kurtosis(f, E_range, index):

  E_min = 0
  E_max = index


  B = quad(f, E_min, E_max, limit = 1000)[0]

  E_prom = quad(lambda x: x * f(x), E_min, E_max, limit = 1000)[0] / B

  sigma_2 = quad(lambda x: f(x) * (x - E_prom)**2 / B, E_min, E_max, limit = 1000)[0]
  sigma = np.sqrt(sigma_2)

  skewness = quad(lambda x: f(x) * ((x - E_prom) / sigma)**3 / (B), E_min, E_max, limit = 1000)[0]

  kurtosis = quad(lambda x: f(x) * ((x - E_prom) / sigma)**4 / (B), E_min, E_max, limit = 1000)[0]

  return skewness, kurtosis

skewness_Mo = []
kurtosis_Mo = []
for i in range(len(fluxes_refined_Mo)):
  skew, kurt = skewness_kurtosis(fluxes_refined_Mo[i], energy_linspaces_Mo[i], 10 + i)
  skewness_Mo.append(skew)
  kurtosis_Mo.append(kurt)

skewness_W = []
kurtosis_W = []
for i in range(len(fluxes_refined_W)):
  skew, kurt = skewness_kurtosis(fluxes_refined_W[i], energy_linspaces_W[i], 10+i)
  skewness_W.append(skew)
  kurtosis_W.append(kurt)

skewness_Rh = []
kurtosis_Rh = []
for i in range(len(fluxes_refined_Rh)):
  skew, kurt = skewness_kurtosis(fluxes_refined_Rh[i], energy_linspaces_Rh[i], 10+i)
  skewness_Rh.append(skew)
  kurtosis_Rh.append(kurt)

voltage = np.arange(10, 50)



"""Análisis de picos"""

# La función baseline_arPLS fue tomada de: Casas-Orozco, D. (Mayo 12, 2021). Baseline correction using arPLS. Stack Overflow. https://stackoverflow.com/questions/29156532/python-baseline-correction-library
# Aclaración: la sección de quitar los picos fue fuertemente influenciada por el código anteriormente referenciado

def baseline_arPLS(y, ratio=1e-6, lam=1e4, niter=20):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio and count < niter:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        if len(dn) == 0:
            break

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m))/s))
        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)
        count += 1

    return z


def process_df_peaks(df, lam=1e5, prominence_frac=0.09, distance=3):
    E = df['energy'].values
    F = df['flux'].values

    baseline = baseline_arPLS(F, lam=lam)
    peaks = F - baseline

    initial_peaks_idx, _ = find_peaks(
        peaks,
        prominence=prominence_frac * np.max(peaks),
        distance=distance
    )

    filtered_peaks_idx = np.copy(initial_peaks_idx)
    if len(initial_peaks_idx) > 2:
        filtered_peaks_idx = initial_peaks_idx[1:]

    return {
        'energy': E,
        'flux': F,
        'baseline': baseline,
        'residual': peaks,
        'peaks_idx': filtered_peaks_idx,
        'energy_peaks': E[filtered_peaks_idx],
        'amplitude_peaks': peaks[filtered_peaks_idx]
    }


def process_df_peaks_W(df, lam=1e5, prominence_frac=0.09, distance=3):
    E = df['energy'].values
    F = df['flux'].values

    baseline = baseline_arPLS(F, lam=lam)
    peaks = F - baseline

    initial_peaks_idx, _ = find_peaks(
        peaks,
        prominence=prominence_frac * np.max(peaks),
        distance=distance
    )

    filtered_peaks_idx = np.copy(initial_peaks_idx)

    return {
        'energy': E,
        'flux': F,
        'baseline': baseline,
        'residual': peaks,
        'peaks_idx': filtered_peaks_idx,
        'energy_peaks': E[filtered_peaks_idx],
        'amplitude_peaks': peaks[filtered_peaks_idx]
    }

def process_element(dataframes, element_name, prominence_frac=0.09, distance=3):
    results = []

    for i, df in enumerate(dataframes):
        out = process_df_peaks(df, prominence_frac=prominence_frac, distance=distance)

        for idx, E_p, A_p in zip(
            out['peaks_idx'],
            out['energy_peaks'],
            out['amplitude_peaks']
        ):
            results.append({
                'element': element_name,
                'file_id': i,
                'voltage': i + 10,
                'peak_index': idx,
                'energy_peak': E_p,
                'amplitude': A_p
            })

    return results


def process_element_W(dataframes, element_name, prominence_frac=0.09, distance=3):
    results = []

    for i, df in enumerate(dataframes):
        out = process_df_peaks_W(df, prominence_frac=prominence_frac, distance=distance)

        for idx, E_p, A_p in zip(
            out['peaks_idx'],
            out['energy_peaks'],
            out['amplitude_peaks']
        ):
            results.append({
                'element': element_name,
                'file_id': i,
                'voltage': i + 10,
                'peak_index': idx,
                'energy_peak': E_p,
                'amplitude': A_p
            })

    return results


results_Mo = process_element(dataframes_Mo, 'Mo')
results_W  = process_element_W(dataframes_W, 'W')
results_Rh = process_element(dataframes_Rh, 'Rh')

df_peaks = pd.DataFrame(results_Mo + results_W + results_Rh)


def plot_spectrum_with_peaks(df, title=""):
    out = process_df_peaks_W(df)

    plt.figure(figsize=(7, 4))
    plt.scatter(out['energy'], out['flux'], label='Espectro original', s=6)
    plt.scatter(
        out['energy_peaks'],
        out['flux'][out['peaks_idx']],
        color='red',
        zorder=3,
        label='Picos detectados'
    )

    plt.xlabel("Energía")
    plt.ylabel("Flujo")
    plt.yscale('asinh')
    plt.title(title)
    plt.legend()
    plt.show()

# mas elegantemente

# La función baseline_arPLS fue tomada de: Casas-Orozco, D. (Mayo 12, 2021). Baseline correction using arPLS. Stack Overflow. https://stackoverflow.com/questions/29156532/python-baseline-correction-library
# Aclaración: la sección de quitar los picos fue fuertemente influenciada por el código anteriormente referenciado

def baseline_arPLS(y, ratio=1e-6, lam=1e4, niter=20):
    L = len(y)

    diag = np.ones(L - 2)
    D = sparse.spdiags([diag, -2*diag, diag], [0, -1, -2], L, L - 2)
    H = lam * D.dot(D.T)

    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)

    crit = 1
    count = 0

    while crit > ratio and count < niter:
        z = linalg.spsolve(W + H, W * y)
        d = y - z
        dn = d[d < 0]

        if len(dn) == 0:
            break

        m = np.mean(dn)
        s = np.std(dn)

        w_new = 1 / (1 + np.exp(2 * (d - (2*s - m)) / s))
        crit = norm(w_new - w) / norm(w)

        w = w_new
        W.setdiag(w)
        count += 1

    return z


def process_df_peaks(df, lam=1e5, prominence_frac=0.09, distance=3):
    E = df["energy"].values
    F = df["flux"].values

    baseline = baseline_arPLS(F, lam=lam)
    residual = F - baseline

    peaks, _ = find_peaks(
        residual,
        prominence=prominence_frac * np.max(residual),
        distance=distance
    )

    return {
        "energy": E,
        "flux": F,
        "residual": residual,
        "peaks_idx": peaks,
        "energy_peaks": E[peaks],
        "amplitude_peaks": residual[peaks]
    }

def process_element(dataframes, element_name, voltage_start=10):
    results = []

    for file_id, df in enumerate(dataframes):
        out = process_df_peaks(df)

        for idx, E_p, A_p in zip(
            out["peaks_idx"],
            out["energy_peaks"],
            out["amplitude_peaks"]
        ):
            results.append({
                "element": element_name,
                "file_id": file_id,
                "voltage": voltage_start + file_id,
                "peak_index": idx,
                "energy_peak": E_p,
                "amplitude": A_p
            })

    return pd.DataFrame(results)

def extract_peak_interval(df, peak_index, interval=3):
    E = df["energy"].values
    F = df["flux"].values

    baseline = baseline_arPLS(F)
    peaks = F - baseline

    i1 = max(0, peak_index - interval)
    i2 = min(len(E), peak_index + interval + 1)

    return pd.DataFrame({
        "energy": E[i1:i2],
        "flux_peak": peaks[i1:i2]
    })





def build_peak_windows(df_peaks, dataframes_dict, interval=3):
    peak_intervals = []

    for _, row in df_peaks.iterrows():
        df = dataframes_dict[row["element"]][row["file_id"]]

        df_peak = extract_peak_interval(df, row["peak_index"], interval)

        df_peak["element"] = row["element"]
        df_peak["file_id"] = row["file_id"]
        df_peak["voltage"] = row["voltage"]
        df_peak["energy_peak"] = row["energy_peak"]

        peak_intervals.append(df_peak)

    return pd.concat(peak_intervals, ignore_index=True)

results_Mo = process_element(dataframes_Mo, "Mo")
results_W  = process_element(dataframes_W,  "W")
results_Rh = process_element(dataframes_Rh, "Rh")

df_peaks = pd.concat([results_Mo, results_W, results_Rh], ignore_index=True)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



def compute_sigma_and_area(df_peaks_windowed):
    results = []

    grouped = df_peaks_windowed.groupby(["element", "file_id", "voltage", "energy_peak"], sort=False)

    for (element, file_id, voltage, energy_peak), df_peak in grouped:
        x = df_peak["energy"].values
        y = df_peak["flux_peak"].values

        A0 = np.max(y)
        mu0 = x[np.argmax(y)]
        sigma0 = np.std(x) if np.std(x) > 0 else 0.1

        popt, _ = curve_fit(gaussian, x, y, p0=[A0, mu0, sigma0], maxfev=5000)

        A, mu, sigma = popt
        sigma = abs(sigma)
        area = A * sigma * np.sqrt(2 * np.pi)

        results.append({"element": element, "file_id": file_id, "voltage": voltage, "energy_peak": energy_peak, "sigma": sigma, "area": area})

    return pd.DataFrame(results)


def plot_gaussian_fits(df_results, df_peaks_windowed, element_name):
    df_elem = df_results[df_results["element"] == element_name]
    df_pts  = df_peaks_windowed[df_peaks_windowed["element"] == element_name]

    for file_id, df_file in df_elem.groupby("file_id"):
        voltage = df_file["voltage"].iloc[0]

        plt.figure(figsize=(7, 4))

        for _, row in df_file.iterrows():
            mu = row["energy_peak"]
            sigma = row["sigma"]
            area = row["area"]

            A = area / (sigma * np.sqrt(2*np.pi))

            x_dense = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
            y_dense = gaussian(x_dense, A, mu, sigma)

            plt.plot(x_dense, y_dense, linewidth=2)

            pts = df_pts[
                (df_pts["file_id"] == file_id) &
                (np.abs(df_pts["energy_peak"] - mu) < 1e-3)
            ]

            plt.scatter(pts["energy"], pts["flux_peak"], s=35)

        plt.xlabel("Energía (keV)")
        plt.ylabel("Flujo (baseline removido)")
        plt.title(f"{element_name} | V = {voltage} kV")
        plt.yscale("log")
        plt.grid(alpha=0.25)
        plt.show()


def process_element(dataframes, element_name, voltage_start=10):
    results = []

    for file_id, df in enumerate(dataframes):
        out = process_df_peaks(df)

        for idx, E_p, A_p in zip(
            out["peaks_idx"],
            out["energy_peaks"],
            out["amplitude_peaks"]
        ):
            results.append({
                "element": element_name,
                "file_id": file_id,
                "voltage": voltage_start + file_id,
                "peak_index": idx,
                "energy_peak": E_p,
                "amplitude": A_p
            })

    return pd.DataFrame(results)


dataframes_dict = {"Mo": dataframes_Mo, "W":  dataframes_W,"Rh": dataframes_Rh}


df_peaks = pd.concat([process_element(dataframes_Mo, "Mo"), process_element(dataframes_W,  "W"), process_element(dataframes_Rh, "Rh")], ignore_index=True)


df_peaks_windowed = build_peak_windows(df_peaks, dataframes_dict)
df_results = compute_sigma_and_area(df_peaks_windowed)
clear_output()


def plot_vs_energy(df_results, quantity, ylabel):

  elements = ["Mo", "W", "Rh"]
  fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey= True)

  for ax, element in zip(axes, elements):
    df_el = df_results[df_results["element"] == element]
    df_el = df_el.sort_values("energy_peak")

    ax.plot(df_el["energy_peak"], df_el[quantity], marker = 'o', linewidth=2)
    ax.set_title(element)
    ax.set_xlabel("Energía del pico (keV)")

  axes[0].set_ylabel(ylabel)
  fig.suptitle(ylabel, fontsize = 14)
  fig.tight_layout()
  plt.show()



df_peaks = pd.concat([process_element(dataframes_Mo, "Mo"), process_element(dataframes_W,  "W"), process_element(dataframes_Rh, "Rh")], ignore_index=True)

dataframes_dict = {"Mo": dataframes_Mo, "W":  dataframes_W, "Rh": dataframes_Rh}

df_peaks_windowed = build_peak_windows(df_peaks, dataframes_dict)

df_results = compute_sigma_and_area(df_peaks_windowed)

clear_output()



def build_peak_indexed_table(df_results, quantity):
    rows = []

    for (element, voltage), df in df_results.groupby(["element", "voltage"]):
        df_sorted = df.sort_values("energy_peak")

        values = df_sorted[quantity].values

        rows.append({
            "element": element,
            "voltage": voltage,
            "P1": values[0] if len(values) > 0 else 0,
            "P2": values[1] if len(values) > 1 else 0,
            "P3": values[2] if len(values) > 2 else 0,
        })

    return pd.DataFrame(rows)


def plot_peaks_vs_voltage(df_results, quantity, ylabel):
    df_plot = build_peak_indexed_table(df_results, quantity)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    peak_cols = ["P1", "P2", "P3"]
    titles = ["Pico 1", "Pico 2", "Pico 3"]
    elements = ["Mo", "W", "Rh"]

    for ax, col, title in zip(axes, peak_cols, titles):
        for element in elements:
            df_el = df_plot[df_plot["element"] == element].sort_values("voltage")

            ax.plot(
                df_el["voltage"],
                df_el[col],
                marker="o",
                linewidth=2,
                label=element
            )

        ax.set_title(title)
        ax.set_xlabel("kV")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(ylabel)
    axes[1].legend(title="Elemento")

    fig.tight_layout()
    plt.show()




df_results = df_results.copy()
df_results["A_gaussian"] = (df_results["area"] / (df_results["sigma"] * np.sqrt(2 * np.pi)))



"""# a"""



total_areas_data = []
voltages = np.arange(10, 50)

for i, f_spline in enumerate(fluxes_refined_Mo):
    V = voltages[i]

    def positive_f_spline_Mo(e):
        return max(0, float(f_spline(e)))

    try:
        total_area, _ = quad(positive_f_spline_Mo, 0, V, limit=500)
        total_areas_data.append({'element': 'Mo', 'voltage': V, 'total_area': total_area})
    except Exception as e:
        print(f"Error integrating Mo for V={V}: {e}")
        total_areas_data.append({'element': 'Mo', 'voltage': V, 'total_area': np.nan})

for i, f_spline in enumerate(fluxes_refined_W):
    V = voltages[i]
    def positive_f_spline_W(e):
        return max(0, float(f_spline(e)))

    try:
        total_area, _ = quad(positive_f_spline_W, 0, V, limit=500)
        total_areas_data.append({'element': 'W', 'voltage': V, 'total_area': total_area})
    except Exception as e:
        print(f"Error integrating W for V={V}: {e}")
        total_areas_data.append({'element': 'W', 'voltage': V, 'total_area': np.nan})


for i, f_spline in enumerate(fluxes_refined_Rh):
    V = voltages[i]
    def positive_f_spline_Rh(e):
        return max(0, float(f_spline(e)))

    try:
        total_area, _ = quad(positive_f_spline_Rh, 0, V, limit=500)
        total_areas_data.append({'element': 'Rh', 'voltage': V, 'total_area': total_area})
    except Exception as e:
        print(f"Error integrating Rh for V={V}: {e}")
        total_areas_data.append({'element': 'Rh', 'voltage': V, 'total_area': np.nan})


df_total_areas = pd.DataFrame(total_areas_data)

print("Total areas DataFrame created:")
print(df_total_areas.head())

df_merged = pd.merge(df_results, df_total_areas, on=['element', 'voltage'], how='left')

print("Merged DataFrame created:")
print(df_merged.head())


df_merged['fractional_area'] = df_merged['area'] / df_merged['total_area']

print("DataFrame with fractional peak areas:")
print(df_merged.head())


plt.figure(figsize=(12, 8))

element_colors = {'Mo': 'cyan', 'W': 'gray', 'Rh': 'red'}

for element, color in element_colors.items():
    df_element = df_merged[df_merged['element'] == element]

    df_element = df_element.sort_values(by='energy_peak')

    for energy_peak_val, df_peak_data in df_element.groupby('energy_peak'):

        label = f"{element} Peak {energy_peak_val:.2f} keV" if element_colors[element] == color else None
        plt.scatter(df_peak_data['voltage'], df_peak_data['fractional_area'], color=color, label=label)



with PdfPages("rayos_X.pdf") as pdf:

    fig, axs = plt.subplots(3, 3, figsize=(22, 14))

    axs[0, 0].plot(energy, max_fluxes_Mo, label='Mo')
    axs[0, 0].scatter(energy, max_fluxes_Mo, s=5)
    axs[0, 0].plot(energy, max_fluxes_W, label='W')
    axs[0, 0].scatter(energy, max_fluxes_W, s=5)
    axs[0, 0].plot(energy, max_fluxes_Rh, label='Rh')
    axs[0, 0].scatter(energy, max_fluxes_Rh, s=5)
    axs[0, 0].set_title('Max Flux')
    axs[0, 0].set_xlabel('Voltage (kV)')
    axs[0, 0].set_ylabel('Max Flux')
    axs[0, 0].grid()
    axs[0, 0].legend()

 
    axs[0, 1].plot(energy, max_energies_Mo, label='Mo')
    axs[0, 1].scatter(energy, max_energies_Mo, s=5)
    axs[0, 1].plot(energy, max_energies_W, label='W')
    axs[0, 1].scatter(energy, max_energies_W, s=5)
    axs[0, 1].plot(energy, max_energies_Rh, label='Rh')
    axs[0, 1].scatter(energy, max_energies_Rh, s=5)
    axs[0, 1].set_title('Energy at Max Flux')
    axs[0, 1].set_xlabel('Voltage (kV)')
    axs[0, 1].set_ylabel('Energy (keV)')
    axs[0, 1].grid()
    axs[0, 1].legend()


    axs[0, 2].plot(skewness_Mo, kurtosis_Mo, label='Mo')
    axs[0, 2].scatter(skewness_Mo, kurtosis_Mo, s=6)
    axs[0, 2].plot(skewness_W, kurtosis_W, label='W')
    axs[0, 2].scatter(skewness_W, kurtosis_W, s=6)
    axs[0, 2].plot(skewness_Rh, kurtosis_Rh, label='Rh')
    axs[0, 2].scatter(skewness_Rh, kurtosis_Rh, s=6)
    axs[0, 2].set_xlabel('Skewness')
    axs[0, 2].set_ylabel('Kurtosis')
    axs[0, 2].set_title('Skewness vs Kurtosis')
    axs[0, 2].grid()
    axs[0, 2].legend()

    axs[1, 0].plot(max_energies_Mo, -np.array(curvature_emax_Mo), label='Mo')
    axs[1, 0].plot(max_energies_W, -np.array(curvature_emax_W), label='W')
    axs[1, 0].plot(max_energies_Rh, -np.array(curvature_emax_Rh), label='Rh')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Curvature')
    axs[1, 0].set_xlabel('Energy (keV)')
    axs[1, 0].set_ylabel('Curvature')
    axs[1, 0].grid()
    axs[1, 0].legend()


    axs[1, 1].plot(range(10, 50), fwhms_Mo, label='Mo')
    axs[1, 1].plot(range(10, 50), fwhms_W, label='W')
    axs[1, 1].plot(range(10, 50), fwhms_Rh, label='Rh')
    axs[1, 1].set_title('FWHM')
    axs[1, 1].set_xlabel('Voltage (kV)')
    axs[1, 1].set_ylabel('FWHM (keV)')
    axs[1, 1].grid()
    axs[1, 1].legend()

    if "A_gaussian" not in df_results.columns:
        df_results["A_gaussian"] = df_results["area"] / (df_results["sigma"] * np.sqrt(2 * np.pi))

    quantities = [
        ("A_gaussian", "Amplitud", axs[1, 2], True),
        ("sigma", "Std Dev", axs[2, 0], False),
        ("area", "Área", axs[2, 1], True),
    ]

    colors = {'Mo': 'blue', 'W': 'green', 'Rh': 'red'}

    for quantity, ylabel, ax, logscale in quantities:
        for element, color in colors.items():
            df_el = df_results[df_results["element"] == element]
            for peak in sorted(df_el["energy_peak"].unique()):
                df_p = df_el[df_el["energy_peak"] == peak]
                ax.plot(df_p["voltage"], df_p[quantity],
                        marker='o', color=color, alpha=0.7,
                        label=f"{element} {peak:.2f} keV")

        ax.set_xlim(21, 50)
        ax.set_xlabel("Voltage (kV)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Voltage")
        ax.grid(True, which="both", ls="--", c='0.7')
        if logscale:
            ax.set_yscale('log')
        ax.legend(fontsize=8)


    axs[2, 2].axis("off") 
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

