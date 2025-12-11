import numpy as np
from scipy.interpolate import RegularGridInterpolator

def genera_superficie(
    L=100.0,
    W=20.0,
    nx=50,
    ny=100,
    kind="plane",
    pendenza_target=0.2,
    amp=2.0,
    freq_x=0.2,
    freq_y=0.2,
):
    """
    Genera una superficie 3D per la pista da sci e restituisce:
    - x_grid, y_grid: vettori 1D dei nodi in x e y
    - Xg, Yg, Zg    : meshgrid della superficie
    - h(x, y)       : funzione interpolante della quota

    Parametri
    ---------
    L : float
        Lunghezza della pista (direzione y).
    W : float
        Larghezza della pista (direzione x).
    nx, ny : int
        Numero di punti griglia in x e y.
    kind : str
        Tipo di superficie:
        - "plane"   : piano inclinato
        - "sinx"    : ondulazioni sinusoidali lungo x
        - "siny"    : ondulazioni sinusoidali lungo y
        - "moguls"  : gobbe bidimensionali
    pendenza_target : float
        Pendenza del piano di base (dz/dy ≈ -pendenza_target).
    amp : float
        Ampiezza delle ondulazioni (per sinx/siny/moguls).
    freq_x, freq_y : float
        Frequenze per le ondulazioni in x e y.
    """

    # Griglia
    x_grid = np.linspace(-W/2, W/2, nx)
    y_grid = np.linspace(0.0, L, ny)
    Xg, Yg = np.meshgrid(x_grid, y_grid)  # shape (ny, nx)


    # Piano base (pendenza costante lungo y)
    Z_base = pendenza_target * L - pendenza_target * Yg # => Z(0) = pendenza_target*L, Z(L) = 0


    # Sovrastruttura a seconda del tipo
    if kind == "plane":
        Zg = Z_base

    elif kind == "sinx":
        # ondulazioni solo lungo x
        Zg = Z_base + amp * np.sin(freq_x * Xg)

    elif kind == "siny":
        # ondulazioni solo lungo y
        Zg = Z_base + amp * np.sin(freq_y * Yg)

    elif kind == "moguls":
        # gobbe bidimensionali (prodotto di seni)
        Zg = Z_base + amp * np.sin(freq_x * Xg) * np.sin(freq_y * Yg)

    else:
        raise ValueError(f"Tipo di superficie sconosciuto: {kind}")

    # Interpolante h(x, y) tramite RegularGridInterpolator
    Fz = RegularGridInterpolator(
        (y_grid, x_grid),
        Zg,
        bounds_error=False,
        fill_value=None  # extrapola linearmente (ok per piccoli delta)
    )

    def h(x, y):
        """
        Restituisce la quota h(x, y).
        x, y possono essere scalari o array (stessa shape).
        """
        pts = np.column_stack([y, x])  # ordine (y, x)
        return Fz(pts)

    return x_grid, y_grid, Xg, Yg, Zg, h
