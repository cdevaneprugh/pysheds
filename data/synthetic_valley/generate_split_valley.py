"""Generate an asymmetric split-valley DEM for testing flow-path vs EDT DTND.

Two V-valleys side by side. Channel B sits 3.01m higher than Channel A,
pushing the drainage divide ~10 columns past the geometric midpoint.
This creates a "divergence zone" where the geographically nearest stream
(Channel B) differs from the hydrologically nearest stream (Channel A).

If compute_hand() were replaced with scipy's distance_transform_edt,
tests on the V-valley would still pass — every pixel's flow-path target
IS the nearest target. This DEM breaks that equivalence.

See data/synthetic_valley/README.md for full geometry and analytical expectations.
"""

import math
import os

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
except ImportError as e:
    raise ImportError(
        "rasterio is required for GeoTIFF output. "
        "Install with: conda install -c conda-forge rasterio"
    ) from e

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NROWS = 200
NCOLS = 1000
PIXEL_SIZE = 5.0  # meters — not 1m, prevents silently passing unit-spacing tests

CROSS_SLOPE = 0.03  # m/m, perpendicular to channel
DOWNSTREAM_SLOPE = 0.001  # m/m, north-to-south along channel
BASE_ELEVATION = 50.0  # meters
CHANNEL_OFFSET = 3.01  # meters — Channel B is this much higher than A

CHANNEL_A_COL = 200
CHANNEL_B_COL = 700

EPSG = 32617
UTM_EASTING = 400000.0
UTM_NORTHING = 3286000.0
NODATA = -9999.0

# Plot output directory
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "..",
    "hpg-esm-tools",
    "swenson",
    "output",
    "plots",
    "synthetic_dem_audit",
)
# Resolve to absolute; fall back to env var if relative traversal fails
PLOT_DIR = os.path.normpath(PLOT_DIR)
if not os.path.isdir(os.path.dirname(PLOT_DIR)):
    PLOT_DIR = os.path.join(
        os.environ.get(
            "SWENSON",
            "/blue/gerber/cdevaneprugh/hpg-esm-tools/swenson",
        ),
        "output",
        "plots",
        "synthetic_dem_audit",
    )


def generate_split_valley() -> tuple[np.ndarray, dict]:
    """Build the split-valley elevation array and analytical expectations.

    Returns
    -------
    elev : np.ndarray
        Elevation array of shape (NROWS, NCOLS), dtype float32.
    expectations : dict
        Analytical expectations for validation.
    """
    rows = np.arange(NROWS, dtype=np.float64)
    cols = np.arange(NCOLS, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    # Two independent V-valley surfaces
    elev_a = (
        BASE_ELEVATION
        + DOWNSTREAM_SLOPE * (NROWS - 1 - rr) * PIXEL_SIZE
        + CROSS_SLOPE * np.abs(cc - CHANNEL_A_COL) * PIXEL_SIZE
    )

    elev_b = (
        BASE_ELEVATION
        + DOWNSTREAM_SLOPE * (NROWS - 1 - rr) * PIXEL_SIZE
        + CROSS_SLOPE * np.abs(cc - CHANNEL_B_COL) * PIXEL_SIZE
        + CHANNEL_OFFSET
    )

    # Combined surface: minimum of the two creates a ridge at intersection
    elev = np.minimum(elev_a, elev_b)
    elev = elev.astype(np.float32)

    # --- Analytical expectations ---

    # Divide location: where elev_a == elev_b
    # cross_slope * |c - 200| * pixel_size = cross_slope * |c - 700| * pixel_size + 3.01
    # Between channels (200 < c < 700):
    #   cs * px * (c - 200) = cs * px * (700 - c) + offset
    #   2 * cs * px * c = cs * px * (200 + 700) + offset
    #   c = (A + B) / 2 + offset / (2 * cs * px)
    divide_col_exact = (CHANNEL_A_COL + CHANNEL_B_COL) / 2.0 + CHANNEL_OFFSET / (
        2 * CROSS_SLOPE * PIXEL_SIZE
    )
    # Integer divide: col <= floor(divide_col_exact) drains to A
    divide_col_int = int(math.floor(divide_col_exact))

    # Geometric midpoint between channels
    midpoint_col = (CHANNEL_A_COL + CHANNEL_B_COL) / 2.0

    # Divergence zone: columns between midpoint and divide
    # These pixels are closer to B but drain to A
    # First col strictly closer to B is midpoint + 1 (since midpoint is equidistant)
    divergence_start = int(math.floor(midpoint_col)) + 1
    divergence_end = divide_col_int  # last col draining to A

    # Slope magnitude (uniform on each planar hillslope)
    slope_mag = math.sqrt(CROSS_SLOPE**2 + DOWNSTREAM_SLOPE**2)

    # HAND for a pixel at column c draining to channel at channel_col:
    # HAND = cross_slope * |c - channel_col| * pixel_size
    # (downstream_slope cancels — same row)

    # Sample DTND values in divergence zone
    divergence_samples = {}
    for c in [divergence_start, 455, divide_col_int]:
        if divergence_start <= c <= divergence_end:
            flow_dtnd = abs(c - CHANNEL_A_COL) * PIXEL_SIZE
            edt_dtnd = abs(c - CHANNEL_B_COL) * PIXEL_SIZE
            divergence_samples[c] = {
                "flow_dtnd": flow_dtnd,
                "edt_dtnd": edt_dtnd,
                "diff": flow_dtnd - edt_dtnd,
            }

    expectations = {
        "nrows": NROWS,
        "ncols": NCOLS,
        "pixel_size": PIXEL_SIZE,
        "cross_slope": CROSS_SLOPE,
        "downstream_slope": DOWNSTREAM_SLOPE,
        "base_elevation": BASE_ELEVATION,
        "channel_offset": CHANNEL_OFFSET,
        "channel_a_col": CHANNEL_A_COL,
        "channel_b_col": CHANNEL_B_COL,
        "divide_col_exact": divide_col_exact,
        "divide_col_int": divide_col_int,
        "midpoint_col": midpoint_col,
        "divergence_start": divergence_start,
        "divergence_end": divergence_end,
        "slope": slope_mag,
        "divergence_samples": divergence_samples,
        "hand_formula": f"HAND = {CROSS_SLOPE} * |c - channel_col| * {PIXEL_SIZE}",
        "dtnd_formula": (
            f"DTND(flow) = |c - channel_col| * {PIXEL_SIZE} "
            "(channel_col depends on which basin)"
        ),
    }

    return elev, expectations


def write_geotiff(elev: np.ndarray, output_path: str) -> None:
    """Write elevation array as a single-band GeoTIFF."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    transform = Affine(PIXEL_SIZE, 0, UTM_EASTING, 0, -PIXEL_SIZE, UTM_NORTHING)
    crs = CRS.from_epsg(EPSG)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=elev.shape[0],
        width=elev.shape[1],
        count=1,
        dtype=elev.dtype,
        crs=crs,
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(elev, 1)


def plot_diagnostics(elev: np.ndarray, expectations: dict) -> None:
    """Generate diagnostic plots for visual inspection."""
    if plt is None:
        print("matplotlib not available — skipping plots.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    mid_row = NROWS // 2
    cols = np.arange(NCOLS)
    col_meters = cols * PIXEL_SIZE

    # --- Plan-view elevation map ---
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(
        elev,
        aspect="auto",
        cmap="terrain",
        extent=[0, NCOLS * PIXEL_SIZE, NROWS * PIXEL_SIZE, 0],
    )
    plt.colorbar(im, ax=ax, label="Elevation (m)")

    # Mark channels and divide
    ax.axvline(
        CHANNEL_A_COL * PIXEL_SIZE, color="blue", ls="--", lw=1.5, label="Channel A"
    )
    ax.axvline(
        CHANNEL_B_COL * PIXEL_SIZE, color="red", ls="--", lw=1.5, label="Channel B"
    )
    ax.axvline(
        expectations["divide_col_exact"] * PIXEL_SIZE,
        color="black",
        ls="-",
        lw=2,
        label=f"Divide (col {expectations['divide_col_exact']:.1f})",
    )
    ax.axvline(
        expectations["midpoint_col"] * PIXEL_SIZE,
        color="gray",
        ls=":",
        lw=1.5,
        label=f"Midpoint (col {expectations['midpoint_col']:.0f})",
    )

    # Shade divergence zone
    ax.axvspan(
        expectations["divergence_start"] * PIXEL_SIZE,
        (expectations["divergence_end"] + 1) * PIXEL_SIZE,
        alpha=0.2,
        color="orange",
        label="Divergence zone",
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing offset (m)")
    ax.set_title("Split Valley — Plan View Elevation")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "split_valley_elevation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")

    # --- Cross-section + expected HAND ---
    rows_arr = np.arange(NROWS, dtype=np.float64)
    cols_arr = np.arange(NCOLS, dtype=np.float64)
    _, cc = np.meshgrid(rows_arr, cols_arr, indexing="ij")

    elev_a_row = (
        BASE_ELEVATION
        + DOWNSTREAM_SLOPE * (NROWS - 1 - mid_row) * PIXEL_SIZE
        + CROSS_SLOPE * np.abs(cols_arr - CHANNEL_A_COL) * PIXEL_SIZE
    )
    elev_b_row = (
        BASE_ELEVATION
        + DOWNSTREAM_SLOPE * (NROWS - 1 - mid_row) * PIXEL_SIZE
        + CROSS_SLOPE * np.abs(cols_arr - CHANNEL_B_COL) * PIXEL_SIZE
        + CHANNEL_OFFSET
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: elevation cross-section
    ax1 = axes[0]
    ax1.plot(col_meters, elev_a_row, "b--", lw=1, alpha=0.7, label="Surface A")
    ax1.plot(col_meters, elev_b_row, "r--", lw=1, alpha=0.7, label="Surface B")
    ax1.plot(col_meters, elev[mid_row, :], "k-", lw=1.5, label="min(A, B)")

    ax1.axvline(CHANNEL_A_COL * PIXEL_SIZE, color="blue", ls=":", lw=1)
    ax1.axvline(CHANNEL_B_COL * PIXEL_SIZE, color="red", ls=":", lw=1)
    ax1.axvline(
        expectations["divide_col_exact"] * PIXEL_SIZE,
        color="black",
        ls="-",
        lw=1.5,
        label="Divide",
    )
    ax1.axvline(
        expectations["midpoint_col"] * PIXEL_SIZE,
        color="gray",
        ls=":",
        lw=1,
        label="Midpoint",
    )
    ax1.axvspan(
        expectations["divergence_start"] * PIXEL_SIZE,
        (expectations["divergence_end"] + 1) * PIXEL_SIZE,
        alpha=0.15,
        color="orange",
        label="Divergence zone",
    )

    ax1.set_ylabel("Elevation (m)")
    ax1.set_title(f"Split Valley Cross-Section (row {mid_row})")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 2: expected HAND
    ax2 = axes[1]
    divide_int = expectations["divide_col_int"]
    hand = np.empty(NCOLS, dtype=np.float64)
    for c in range(NCOLS):
        if c <= divide_int:
            hand[c] = CROSS_SLOPE * abs(c - CHANNEL_A_COL) * PIXEL_SIZE
        else:
            hand[c] = CROSS_SLOPE * abs(c - CHANNEL_B_COL) * PIXEL_SIZE

    ax2.plot(col_meters, hand, "k-", lw=1.5, label="Expected HAND (flow-path)")

    # Show what EDT-based HAND would give
    edt_hand = np.minimum(
        CROSS_SLOPE * np.abs(cols_arr - CHANNEL_A_COL) * PIXEL_SIZE,
        CROSS_SLOPE * np.abs(cols_arr - CHANNEL_B_COL) * PIXEL_SIZE,
    )
    ax2.plot(
        col_meters, edt_hand, "r--", lw=1, alpha=0.7, label="EDT-based HAND (wrong)"
    )

    ax2.axvline(
        expectations["divide_col_exact"] * PIXEL_SIZE,
        color="black",
        ls="-",
        lw=1.5,
    )
    ax2.axvline(
        expectations["midpoint_col"] * PIXEL_SIZE,
        color="gray",
        ls=":",
        lw=1,
    )
    ax2.axvspan(
        expectations["divergence_start"] * PIXEL_SIZE,
        (expectations["divergence_end"] + 1) * PIXEL_SIZE,
        alpha=0.15,
        color="orange",
    )

    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("HAND (m)")
    ax2.set_title("Expected HAND — Flow-Path vs EDT")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "split_valley_cross_section.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


def print_summary(expectations: dict) -> None:
    """Print geometry and analytical expectations."""
    print("=" * 70)
    print("Asymmetric Split Valley DEM")
    print("=" * 70)

    print("\nGrid:")
    print(f"  Size:        {NROWS} x {NCOLS} pixels")
    print(f"  Pixel size:  {PIXEL_SIZE} m")
    print(f"  Domain:      {NROWS * PIXEL_SIZE:.0f} x {NCOLS * PIXEL_SIZE:.0f} m")
    print(f"  CRS:         EPSG:{EPSG}")

    print("\nGeometry:")
    print(f"  Cross-slope:       {CROSS_SLOPE} m/m")
    print(f"  Downstream slope:  {DOWNSTREAM_SLOPE} m/m")
    print(f"  Base elevation:    {BASE_ELEVATION} m")
    print(f"  Channel A:         col {CHANNEL_A_COL}")
    print(f"  Channel B:         col {CHANNEL_B_COL} (offset +{CHANNEL_OFFSET} m)")

    print("\nDrainage divide:")
    print(f"  Exact:       col {expectations['divide_col_exact']:.2f}")
    print(
        f"  Integer:     col {expectations['divide_col_int']} (last col draining to A)"
    )
    print(f"  Midpoint:    col {expectations['midpoint_col']:.0f} (geometric center)")

    print("\nDivergence zone (closer to B, drains to A):")
    print(
        f"  Columns:     {expectations['divergence_start']} to {expectations['divergence_end']}"
    )
    n_div = expectations["divergence_end"] - expectations["divergence_start"] + 1
    print(f"  Width:       {n_div} columns ({n_div * PIXEL_SIZE:.0f} m)")
    for c, vals in sorted(expectations["divergence_samples"].items()):
        print(
            f"  Col {c}:     flow DTND = {vals['flow_dtnd']:.0f}m, "
            f"EDT DTND = {vals['edt_dtnd']:.0f}m, "
            f"diff = {vals['diff']:.0f}m"
        )

    print(f"\nSlope: {expectations['slope']:.6f} m/m (uniform on each hillslope)")
    print(f"  {expectations['hand_formula']}")
    print(f"  {expectations['dtnd_formula']}")

    print("=" * 70)


def main() -> None:
    """Generate split-valley DEM, write GeoTIFF, produce plots."""
    output_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "split_valley_utm.tif",
    )

    elev, expectations = generate_split_valley()

    write_geotiff(elev, output_path)
    print(f"Wrote: {output_path}")

    print_summary(expectations)

    # Spot checks — verify the divide is a ridge
    print("\nSpot checks:")
    divide = expectations["divide_col_int"]
    mid_row = NROWS // 2
    print(
        f"  elev[{mid_row}, {divide}]   = {elev[mid_row, divide]:.4f} m (last col in basin A)"
    )
    print(
        f"  elev[{mid_row}, {divide + 1}] = {elev[mid_row, divide + 1]:.4f} m (first col in basin B)"
    )

    # At the divide col, flow should route WEST (toward A): westward neighbor is lower
    routes_west_at_divide = np.all(elev[:, divide - 1] < elev[:, divide])
    print(f"  Col {divide} routes west (toward A): {routes_west_at_divide}")

    # At divide+1, flow should route EAST (toward B): eastward neighbor is lower
    routes_east_past_divide = np.all(elev[:, divide + 2] < elev[:, divide + 1])
    print(f"  Col {divide + 1} routes east (toward B): {routes_east_past_divide}")

    # The ridge: col divide is higher than col divide+1 (A-side peak > B-side start)
    # This is expected because surface A at the divide is just barely below surface B
    print(
        f"  Ridge at divide: elev[{divide}] > elev[{divide + 1}]? "
        f"{np.all(elev[:, divide] > elev[:, divide + 1])}"
    )

    print("\nPlots:")
    plot_diagnostics(elev, expectations)


if __name__ == "__main__":
    main()
