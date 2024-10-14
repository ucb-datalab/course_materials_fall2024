import marimo

__generated_with = "0.9.9"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.constants import c as C, h, k as k_B

    import pandas as pd
    import micropip
    import json
    #await micropip.install('altair')
    #await micropip.install('pyodide')
    import altair as alt

    import warnings
    warnings.filterwarnings("ignore")
    return C, alt, h, json, k_B, micropip, np, pd, plt, warnings


@app.cell
def __(C, alt, gains, h, k_B, mo, np, pd, slider):
    wavelens = np.linspace(15000, 17000, 2001)
    chip1 = (wavelens > 15150) & (wavelens < 15800)
    chip2 = (wavelens > 15890) & (wavelens < 16430)
    chip3 = (wavelens > 16490) & (wavelens < 16950)
    off_chip = ~(chip1 | chip2 | chip3)
    ANGSTROM = 1e-10
    wavelens = wavelens * ANGSTROM

    N_LINES = slider.value

    line_centers = np.random.uniform(wavelens[0], wavelens[-1], size=N_LINES)
    taus = np.random.uniform(0, 3, size=N_LINES)

    def planck(T, wavelens):
        return 2 * h * C ** 2 / wavelens ** 5 / (np.exp(h * C / (wavelens * k_B * T)) - 1)

    star_spec = planck(6000, wavelens)
    line_width = 1.5 * ANGSTROM
    tau_lines = taus[:, None] * (wavelens[None, :] - line_centers[:, None]) ** 2 / (2 * line_width ** 2)
    abs_lines = np.product(1 - np.exp(-tau_lines), axis=0)
    gain_freq = gains.value
    gain = 1e-13 * (1 + 0.1 * np.cos(2 * np.pi * wavelens / (gain_freq * ANGSTROM)))
    noise = np.random.normal(size=wavelens.size) * 1e-06
    meas_spec = -2.5 * np.log10(gain * (star_spec * abs_lines) + noise)
    meas_spec[off_chip] = 10
    #plt.figure()
    #plt.plot(wavelens / ANGSTROM, meas_spec)
    #plt.ylim(5, -1)
    #plt.show()


    data = pd.DataFrame({
        'wavelength (Å)': wavelens / ANGSTROM,
        'measurement': meas_spec
    })

    chart = alt.Chart(data).mark_line().encode(
        x='wavelength (Å)',
        y=alt.Y('measurement', scale=alt.Scale(domain=[5, -0.1]))
    ).properties(
        width=600,
        height=400
    )

    chart = mo.ui.altair_chart(chart)
    chart
    return (
        ANGSTROM,
        N_LINES,
        abs_lines,
        chart,
        chip1,
        chip2,
        chip3,
        data,
        gain,
        gain_freq,
        line_centers,
        line_width,
        meas_spec,
        noise,
        off_chip,
        planck,
        star_spec,
        tau_lines,
        taus,
        wavelens,
    )


@app.cell
def __(mo):
    slider = mo.ui.slider(start=10, stop=300, step=1, label="NLINES")
    gains = mo.ui.slider(start=950, stop=1050, step=5, label="gain frequency")
    return gains, slider


@app.cell
def __(gains, mo, slider):
    mo.vstack( [mo.hstack([slider, mo.md(f"NLINES: {slider.value}")]),
                mo.hstack([gains, mo.md(f"gain freq: {gains.value}")])])
    return


@app.cell
def __(np, plt):
    def px0(a, b):
        return 0.1 + 0.05 * a - 0.05 * b

    def px1(a, b):
        return -0.5 - 0.1 * a - 0.11 * b

    def px2(a, b):
        return 0.9 * (a - 2)**2 - 0.3 * (b - 6)**2

    a0, b0 = 2, 6
    mini_spec = np.array([px0(a0, b0), px1(a0, b0), px2(a0, b0)])

    plt.figure()
    _ = plt.plot(mini_spec)
    plt.show()
    return a0, b0, mini_spec, px0, px1, px2


@app.cell
def __(np, plt, px0, px1, px2):
    a, b = (np.linspace(1, 3, 10), np.linspace(5, 10, 20))
    a, b = np.meshgrid(a, b)
    mini_spec_1 = np.array([px0(a, b), px1(a, b), px2(a, b)])
    plt.figure()
    _ = plt.plot(mini_spec_1.reshape(3, -1), '.')
    plt.show()
    return a, b, mini_spec_1


@app.cell
def __(a, b, mini_spec_1, plt):
    fig, axes = plt.subplots(subplot_kw={'projection': '3d'})
    surf = axes.plot_surface(a, b, mini_spec_1[2], rstride=1, cstride=1, cmap='seismic', linewidth=0, antialiased=False)

    plt.show()
    return axes, fig, surf


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
