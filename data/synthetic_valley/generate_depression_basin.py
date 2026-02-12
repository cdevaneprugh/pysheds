"""Generate a depression basin DEM for testing fill_depressions and resolve_flats.

A north-to-south tilted plane with a parabolic bowl depression near the center.
After fill_depressions(), the bowl becomes a flat at the spill elevation.
After resolve_flats(), the Garbrecht-Martz algorithm should assign micro-gradients
routing flow across the flat toward the spill point.

This is the first synthetic DEM that requires the complete
fill_depressions -> resolve_flats -> flowdir chain. The V-valley and split
valley both skip those steps entirely.

See data/synthetic_valley/README.md for full geometry and analytical expectations.
"""

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
NCOLS = 200
PIXEL_SIZE = 5.0  # meters

BACKGROUND_SLOPE = 0.01  # m/m, north-to-south tilt
BASE_ELEVATION = 50.0  # meters, elevation at the southern edge (row 199)

# Depression parameters
DEP_CENTER_ROW = 100
DEP_CENTER_COL = 100
DEP_RADIUS_PX = 30  # pixels (150m physical)
DEP_DEPTH = 3.0  # meters, maximum depth below background

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


def generate_depression_basin() -> tuple[np.ndarray, dict]:
    """Build the depression basin elevation array and analytical expectations.

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

    # Background: uniform north-to-south tilt
    # Row 0 (north) is highest, row NROWS-1 (south) is lowest
    background = BASE_ELEVATION + BACKGROUND_SLOPE * (NROWS - 1 - rr) * PIXEL_SIZE

    # Parabolic depression: depth = DEP_DEPTH * (1 - (dist/radius)^2)
    dist_px = np.sqrt((rr - DEP_CENTER_ROW) ** 2 + (cc - DEP_CENTER_COL) ** 2)
    depression_mask = dist_px < DEP_RADIUS_PX
    depth = np.zeros_like(background)
    depth[depression_mask] = DEP_DEPTH * (
        1.0 - (dist_px[depression_mask] / DEP_RADIUS_PX) ** 2
    )

    elev = background - depth
    elev = elev.astype(np.float32)

    # --- Analytical expectations ---

    # Background elevation at depression center
    bg_at_center = (
        BASE_ELEVATION + BACKGROUND_SLOPE * (NROWS - 1 - DEP_CENTER_ROW) * PIXEL_SIZE
    )
    dep_center_elev = bg_at_center - DEP_DEPTH

    # Spill point: lowest elevation on the discrete rim of the depression.
    # The rim is the set of pixels just outside the depression that have at
    # least one 8-connected neighbor inside it.
    inside = depression_mask
    # Pad to handle edges, then check all 8 neighbors
    padded = np.pad(inside, 1, mode="constant", constant_values=False)
    has_inside_neighbor = np.zeros_like(inside)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            has_inside_neighbor |= padded[
                1 + dr : NROWS + 1 + dr, 1 + dc : NCOLS + 1 + dc
            ]
    rim_mask = ~inside & has_inside_neighbor

    # Find the rim pixel with the lowest elevation (using float32 elev)
    rim_elevs = np.where(rim_mask, elev, np.inf)
    spill_idx = np.argmin(rim_elevs)
    spill_row, spill_col = np.unravel_index(spill_idx, elev.shape)
    spill_elev = float(elev[spill_row, spill_col])

    # Northern rim (highest rim point)
    north_rim_row = DEP_CENTER_ROW - DEP_RADIUS_PX
    north_rim_elev = (
        BASE_ELEVATION + BACKGROUND_SLOPE * (NROWS - 1 - north_rim_row) * PIXEL_SIZE
    )

    # Number of pixels inside the depression
    n_depression_px = int(np.sum(depression_mask))

    # Number of pixels below spill elevation (these get filled)
    below_spill = (elev < spill_elev) & depression_mask
    n_filled_px = int(np.sum(below_spill))

    # After filling: all pixels below spill_elev are raised to spill_elev
    # This creates a flat region at spill_elev

    # Depression area in physical units
    dep_area_m2 = n_depression_px * PIXEL_SIZE**2
    filled_area_m2 = n_filled_px * PIXEL_SIZE**2

    expectations = {
        "nrows": NROWS,
        "ncols": NCOLS,
        "pixel_size": PIXEL_SIZE,
        "background_slope": BACKGROUND_SLOPE,
        "base_elevation": BASE_ELEVATION,
        "dep_center": (DEP_CENTER_ROW, DEP_CENTER_COL),
        "dep_radius_px": DEP_RADIUS_PX,
        "dep_radius_m": DEP_RADIUS_PX * PIXEL_SIZE,
        "dep_depth": DEP_DEPTH,
        "bg_at_center": bg_at_center,
        "dep_center_elev": dep_center_elev,
        "spill_point": (spill_row, spill_col),
        "spill_elev": spill_elev,
        "north_rim_elev": north_rim_elev,
        "n_depression_px": n_depression_px,
        "n_filled_px": n_filled_px,
        "dep_area_m2": dep_area_m2,
        "filled_area_m2": filled_area_m2,
        "elev_range": (float(np.min(elev)), float(np.max(elev))),
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

    # --- Plan-view elevation map ---
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        elev,
        cmap="terrain",
        extent=[0, NCOLS * PIXEL_SIZE, NROWS * PIXEL_SIZE, 0],
    )
    plt.colorbar(im, ax=ax, label="Elevation (m)")

    # Mark depression outline
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = (DEP_CENTER_COL + DEP_RADIUS_PX * np.cos(theta)) * PIXEL_SIZE
    circle_y = (DEP_CENTER_ROW + DEP_RADIUS_PX * np.sin(theta)) * PIXEL_SIZE
    ax.plot(circle_x, circle_y, "k--", lw=1.5, label="Depression rim")

    # Mark spill point
    sp = expectations["spill_point"]
    ax.plot(sp[1] * PIXEL_SIZE, sp[0] * PIXEL_SIZE, "rv", ms=10, label="Spill point")

    # Mark depression center
    ax.plot(
        DEP_CENTER_COL * PIXEL_SIZE,
        DEP_CENTER_ROW * PIXEL_SIZE,
        "k+",
        ms=12,
        mew=2,
        label="Depression center",
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing offset (m)")
    ax.set_title("Depression Basin — Plan View Elevation")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "depression_basin_elevation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")

    # --- Cross-sections ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel 1: N-S cross-section through depression center (col = DEP_CENTER_COL)
    ax1 = axes[0]
    row_meters = np.arange(NROWS) * PIXEL_SIZE
    ns_profile = elev[:, DEP_CENTER_COL]
    ns_background = (
        BASE_ELEVATION + BACKGROUND_SLOPE * (NROWS - 1 - np.arange(NROWS)) * PIXEL_SIZE
    )

    ax1.plot(
        row_meters, ns_background, "b--", lw=1, alpha=0.7, label="Background surface"
    )
    ax1.plot(row_meters, ns_profile, "k-", lw=1.5, label="Actual elevation")
    ax1.axhline(
        expectations["spill_elev"],
        color="red",
        ls=":",
        lw=1.5,
        label=f"Spill elevation ({expectations['spill_elev']:.2f} m)",
    )

    # Mark spill point
    ax1.plot(
        sp[0] * PIXEL_SIZE,
        expectations["spill_elev"],
        "rv",
        ms=8,
        label="Spill point",
    )

    ax1.set_xlabel("Row distance from north (m)")
    ax1.set_ylabel("Elevation (m)")
    ax1.set_title(f"N-S Cross-Section (col {DEP_CENTER_COL})")
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: E-W cross-section through depression center (row = DEP_CENTER_ROW)
    ax2 = axes[1]
    col_meters = np.arange(NCOLS) * PIXEL_SIZE
    ew_profile = elev[DEP_CENTER_ROW, :]
    ew_background_val = (
        BASE_ELEVATION + BACKGROUND_SLOPE * (NROWS - 1 - DEP_CENTER_ROW) * PIXEL_SIZE
    )

    ax2.axhline(
        ew_background_val, color="b", ls="--", lw=1, alpha=0.7, label="Background level"
    )
    ax2.plot(col_meters, ew_profile, "k-", lw=1.5, label="Actual elevation")
    ax2.axhline(
        expectations["spill_elev"],
        color="red",
        ls=":",
        lw=1.5,
        label=f"Spill elevation ({expectations['spill_elev']:.2f} m)",
    )

    ax2.set_xlabel("Column distance from west (m)")
    ax2.set_ylabel("Elevation (m)")
    ax2.set_title(f"E-W Cross-Section (row {DEP_CENTER_ROW})")
    ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "depression_basin_cross_section.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot: {path}")


def print_summary(expectations: dict) -> None:
    """Print geometry and analytical expectations."""
    print("=" * 70)
    print("Depression Basin DEM")
    print("=" * 70)

    print("\nGrid:")
    print(f"  Size:        {NROWS} x {NCOLS} pixels")
    print(f"  Pixel size:  {PIXEL_SIZE} m")
    print(f"  Domain:      {NROWS * PIXEL_SIZE:.0f} x {NCOLS * PIXEL_SIZE:.0f} m")
    print(f"  CRS:         EPSG:{EPSG}")

    print("\nBackground surface:")
    print(f"  Slope:       {BACKGROUND_SLOPE} m/m (north-to-south)")
    print(f"  Base elev:   {BASE_ELEVATION} m (at southern edge)")
    print(f"  North edge:  {expectations['elev_range'][1]:.2f} m")
    print(f"  South edge:  {BASE_ELEVATION:.2f} m")

    print("\nDepression:")
    print(f"  Center:      ({DEP_CENTER_ROW}, {DEP_CENTER_COL})")
    print(f"  Radius:      {DEP_RADIUS_PX} px ({expectations['dep_radius_m']:.0f} m)")
    print(f"  Max depth:   {DEP_DEPTH} m (parabolic profile)")
    print(f"  Pixels:      {expectations['n_depression_px']}")
    print(f"  Area:        {expectations['dep_area_m2']:.0f} m^2")

    print("\nKey elevations:")
    print(f"  Background at center: {expectations['bg_at_center']:.2f} m")
    print(f"  Depression center:    {expectations['dep_center_elev']:.2f} m")
    print(
        f"  Spill point:          {expectations['spill_elev']:.2f} m "
        f"(row {expectations['spill_point'][0]}, col {expectations['spill_point'][1]})"
    )
    print(f"  North rim:            {expectations['north_rim_elev']:.2f} m")

    print("\nAfter fill_depressions:")
    print(f"  Pixels to fill:   {expectations['n_filled_px']}")
    print(f"  Filled area:      {expectations['filled_area_m2']:.0f} m^2")
    print(f"  Flat elevation:   {expectations['spill_elev']:.2f} m")

    print("\nAfter resolve_flats:")
    print("  All flat pixels should receive micro-gradients toward spill point")
    print("  No flat-coded flow directions should remain")

    print("\nAfter flowdir + accumulation:")
    print("  Every pixel should have a valid D8 direction")
    print(
        f"  Spill point accumulation >= {expectations['n_filled_px']} pixels "
        f"(entire filled area drains through it)"
    )
    print("  HAND and DTND should be finite and non-negative everywhere")

    print("=" * 70)


def main() -> None:
    """Generate depression basin DEM, write GeoTIFF, produce plots."""
    output_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "depression_basin_utm.tif",
    )

    elev, expectations = generate_depression_basin()

    write_geotiff(elev, output_path)
    print(f"Wrote: {output_path}")

    print_summary(expectations)

    # Spot checks
    print("\nSpot checks:")
    print(
        f"  elev[{DEP_CENTER_ROW}, {DEP_CENTER_COL}] (center): {elev[DEP_CENTER_ROW, DEP_CENTER_COL]:.4f} m"
    )
    sp = expectations["spill_point"]
    print(f"  elev[{sp[0]}, {sp[1]}] (spill):  {elev[sp[0], sp[1]]:.4f} m")
    print(f"  min(elev inside depression):   {expectations['dep_center_elev']:.4f} m")
    print(
        f"  Depression center < spill:     {expectations['dep_center_elev'] < expectations['spill_elev']}"
    )

    # Verify no flat pixels outside the depression (background is uniformly tilted)
    bg_only = elev.copy()
    dist_px = np.sqrt(
        (np.arange(NROWS)[:, None] - DEP_CENTER_ROW) ** 2
        + (np.arange(NCOLS)[None, :] - DEP_CENTER_COL) ** 2
    )
    outside = dist_px >= DEP_RADIUS_PX
    # Check row-wise differences (should all be negative = decreasing southward)
    row_diffs = np.diff(bg_only[:, DEP_CENTER_COL])
    all_decreasing = np.all(row_diffs[outside[:-1, DEP_CENTER_COL]] < 0)
    print(f"  Background strictly decreasing S: {all_decreasing}")

    print("\nPlots:")
    plot_diagnostics(elev, expectations)


if __name__ == "__main__":
    main()
