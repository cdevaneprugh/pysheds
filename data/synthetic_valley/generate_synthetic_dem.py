"""Generate a synthetic V-valley DEM for testing pysheds UTM CRS handling.

Creates a V-shaped valley with two planar hillslopes meeting at a central
north-south channel. Every output parameter (slope, aspect, HAND, DTND) is
analytically computable from the input geometry, enabling exact validation
of pysheds' gradient, distance, and flow routing computations on UTM data.

The existing test DEM (data/dem.tif) is geographic CRS (EPSG:4326, ~93m
resolution). This synthetic DEM provides a UTM counterpart (EPSG:32617, 1m)
to exercise the UTM code path that exposed CRS bugs in compute_hand() and
_gradient_horn_1981().

See data/synthetic_valley/README.md for full scientific rationale and
analytical expectations.
"""

import argparse
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


def generate_v_valley(
    nrows: int = 1000,
    ncols: int = 1000,
    pixel_size: float = 1.0,
    cross_slope: float = 0.03,
    downstream_slope: float = 0.001,
    base_elevation: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """Generate a V-valley elevation array with analytical expectations.

    The valley has two planar hillslopes meeting at a central north-south
    channel. Row 0 is the north (upstream) end; row nrows-1 is the south
    (downstream) end. The channel runs along the center column.

    Elevation formula:
        elev[r, c] = base_elevation
                   + downstream_slope * (nrows - 1 - r) * pixel_size
                   + cross_slope * abs(c - channel_col) * pixel_size

    Parameters
    ----------
    nrows : int
        Number of rows (north-south extent).
    ncols : int
        Number of columns (east-west extent).
    pixel_size : float
        Pixel size in meters.
    cross_slope : float
        Cross-valley slope (m/m). Gradient perpendicular to channel.
    downstream_slope : float
        Along-channel slope (m/m). North-to-south gradient.
    base_elevation : float
        Elevation at the downstream end of the channel (meters).

    Returns
    -------
    elev : np.ndarray
        Elevation array of shape (nrows, ncols), dtype float32.
    expectations : dict
        Analytical expectations for validation:
        - slope: expected gradient magnitude (m/m)
        - slope_degrees: expected slope in degrees
        - aspect_west_side: aspect of west-side pixels (degrees, clockwise from N)
        - aspect_east_side: aspect of east-side pixels (degrees)
        - aspect_channel: aspect of channel pixels (degrees)
        - hand_max: maximum HAND value (meters)
        - hand_formula: string describing HAND as function of column offset
        - dtnd_max: maximum DTND value (meters)
        - dtnd_formula: string describing DTND as function of column offset
        - channel_col: column index of the channel
        - channel_elev_north: elevation at upstream end of channel
        - channel_elev_south: elevation at downstream end of channel
        - ridge_elev_max: maximum elevation (upstream ridge corner)
    """
    channel_col = ncols // 2

    # Build row and column index arrays
    rows = np.arange(nrows, dtype=np.float64)
    cols = np.arange(ncols, dtype=np.float64)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    # Elevation formula
    elev = (
        base_elevation
        + downstream_slope * (nrows - 1 - rr) * pixel_size
        + cross_slope * np.abs(cc - channel_col) * pixel_size
    )

    elev = elev.astype(np.float32)

    # Analytical expectations
    slope_mag = math.sqrt(cross_slope**2 + downstream_slope**2)
    slope_deg = math.degrees(math.atan(slope_mag))

    # Aspect: direction of steepest descent (clockwise from north)
    # Computed via arctan2(east_downhill, north_downhill) to handle all quadrants correctly.
    #
    # West side (c < channel_col): drains east (+x) and south (-north)
    aspect_west = math.degrees(math.atan2(cross_slope, -downstream_slope))
    # East side (c > channel_col): drains west (-x) and south (-north)
    aspect_east = math.degrees(math.atan2(-cross_slope, -downstream_slope))
    if aspect_east < 0:
        aspect_east += 360.0
    # Channel: only downstream component
    aspect_channel = 180.0

    # Buggy aspects (from pipeline's arctan2(-dzdx, -dzdy) which inverts the north component)
    buggy_aspect_west = math.degrees(math.atan2(cross_slope, downstream_slope))
    buggy_aspect_east = math.degrees(math.atan2(-cross_slope, downstream_slope))
    if buggy_aspect_east < 0:
        buggy_aspect_east += 360.0

    # HAND and DTND
    half_width = (ncols // 2) * pixel_size
    hand_max = cross_slope * half_width
    dtnd_max = half_width

    # Key elevations
    channel_elev_south = base_elevation
    channel_elev_north = base_elevation + downstream_slope * (nrows - 1) * pixel_size
    ridge_elev_max = channel_elev_north + cross_slope * half_width

    expectations = {
        "slope": slope_mag,
        "slope_degrees": slope_deg,
        "aspect_west_side": aspect_west,
        "aspect_east_side": aspect_east,
        "aspect_channel": aspect_channel,
        "buggy_aspect_west_side": buggy_aspect_west,
        "buggy_aspect_east_side": buggy_aspect_east,
        "aspect_error": abs(aspect_west - buggy_aspect_west),
        "hand_max": hand_max,
        "hand_formula": f"HAND = {cross_slope} * abs(c - {channel_col}) * {pixel_size}",
        "dtnd_max": dtnd_max,
        "dtnd_formula": f"DTND = abs(c - {channel_col}) * {pixel_size}",
        "channel_col": channel_col,
        "channel_elev_south": channel_elev_south,
        "channel_elev_north": channel_elev_north,
        "ridge_elev_max": ridge_elev_max,
        "cross_slope": cross_slope,
        "downstream_slope": downstream_slope,
        "pixel_size": pixel_size,
        "nrows": nrows,
        "ncols": ncols,
        "base_elevation": base_elevation,
        "channel_length": (nrows - 1) * pixel_size,
        "channel_elev_drop": downstream_slope * (nrows - 1) * pixel_size,
        "channel_slope": downstream_slope,
        "num_profiles": 1,
    }

    return elev, expectations


def write_geotiff(
    elev: np.ndarray,
    transform: Affine,
    crs: CRS,
    nodata: float,
    output_path: str,
) -> None:
    """Write elevation array as a single-band GeoTIFF.

    Parameters
    ----------
    elev : np.ndarray
        2D elevation array (nrows, ncols), dtype float32.
    transform : Affine
        Affine transform mapping pixel coordinates to CRS coordinates.
    crs : CRS
        Coordinate reference system.
    nodata : float
        Nodata value.
    output_path : str
        Output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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
        nodata=nodata,
    ) as dst:
        dst.write(elev, 1)


def print_summary(params: dict, expectations: dict) -> None:
    """Print grid metadata, parameter values, and analytical expectations."""
    print("=" * 70)
    print("Synthetic V-Valley DEM")
    print("=" * 70)

    print("\nGrid:")
    print(f"  Size:        {expectations['nrows']} x {expectations['ncols']} pixels")
    print(f"  Pixel size:  {expectations['pixel_size']} m")
    print(f"  CRS:         EPSG:{params['epsg']}")
    print(f"  Easting:     {params['utm_easting']}")
    print(f"  Northing:    {params['utm_northing']}")

    print("\nGeometry:")
    print(f"  Cross-slope:       {expectations['cross_slope']} m/m")
    print(f"  Downstream slope:  {expectations['downstream_slope']} m/m")
    print(f"  Base elevation:    {expectations['base_elevation']} m")
    print(f"  Channel column:    {expectations['channel_col']}")

    print("\nElevation range:")
    print(f"  Channel south end: {expectations['channel_elev_south']:.3f} m")
    print(f"  Channel north end: {expectations['channel_elev_north']:.3f} m")
    print(f"  Ridge max:         {expectations['ridge_elev_max']:.3f} m")

    print("\nAnalytical expectations:")
    print(
        f"  Slope:        {expectations['slope']:.6f} m/m ({expectations['slope_degrees']:.4f} deg)"
    )
    print(f"  Aspect west:  {expectations['aspect_west_side']:.2f} deg (east-facing)")
    print(f"  Aspect east:  {expectations['aspect_east_side']:.2f} deg (west-facing)")
    print(f"  Aspect chan:  {expectations['aspect_channel']:.2f} deg (south-facing)")
    print(f"  HAND max:     {expectations['hand_max']:.2f} m (at ridges)")
    print(f"  DTND max:     {expectations['dtnd_max']:.2f} m (at ridges)")
    print(f"  {expectations['hand_formula']}")
    print(f"  {expectations['dtnd_formula']}")

    print("\nBug detection (pipeline aspect sign bug):")
    print(
        f"  West side: correct={expectations['aspect_west_side']:.2f} deg, "
        f"buggy={expectations['buggy_aspect_west_side']:.2f} deg "
        f"(error={expectations['aspect_error']:.2f} deg)"
    )
    print(
        f"  East side: correct={expectations['aspect_east_side']:.2f} deg, "
        f"buggy={expectations['buggy_aspect_east_side']:.2f} deg "
        f"(error={expectations['aspect_error']:.2f} deg)"
    )
    print(
        f"  Error is small ({expectations['aspect_error']:.1f} deg) because cross-slope "
        f"({expectations['cross_slope']}) >> downstream slope ({expectations['downstream_slope']})"
    )
    print("\nBug detection (haversine on UTM):")
    print(
        f"  DTND should be {expectations['dtnd_max']:.1f} m at ridge, "
        "not garbage from haversine on meter-valued coords"
    )

    print("=" * 70)


def main() -> None:
    """Parse CLI arguments, generate V-valley DEM, write GeoTIFF."""
    default_output = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "synthetic_valley_utm.tif",
    )

    parser = argparse.ArgumentParser(
        description="Generate a synthetic V-valley DEM for testing pysheds UTM CRS handling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nrows", type=int, default=1000, help="Number of rows")
    parser.add_argument("--ncols", type=int, default=1000, help="Number of columns")
    parser.add_argument(
        "--pixel-size", type=float, default=1.0, help="Pixel size in meters"
    )
    parser.add_argument(
        "--cross-slope", type=float, default=0.03, help="Cross-valley slope (m/m)"
    )
    parser.add_argument(
        "--downstream-slope",
        type=float,
        default=0.001,
        help="Along-channel slope (m/m)",
    )
    parser.add_argument(
        "--base-elevation", type=float, default=50.0, help="Base elevation (m)"
    )
    parser.add_argument(
        "--utm-easting", type=float, default=400000, help="UTM easting of NW corner"
    )
    parser.add_argument(
        "--utm-northing", type=float, default=3286000, help="UTM northing of NW corner"
    )
    parser.add_argument("--epsg", type=int, default=32617, help="EPSG code for CRS")
    parser.add_argument(
        "--output", type=str, default=default_output, help="Output GeoTIFF path"
    )

    args = parser.parse_args()

    # Generate elevation array and expectations
    elev, expectations = generate_v_valley(
        nrows=args.nrows,
        ncols=args.ncols,
        pixel_size=args.pixel_size,
        cross_slope=args.cross_slope,
        downstream_slope=args.downstream_slope,
        base_elevation=args.base_elevation,
    )

    # Build affine transform: row 0 = north edge, y decreases southward
    transform = Affine(
        args.pixel_size,
        0,
        args.utm_easting,
        0,
        -args.pixel_size,
        args.utm_northing,
    )

    crs = CRS.from_epsg(args.epsg)
    nodata = -9999.0

    # CLI params for summary
    params = {
        "epsg": args.epsg,
        "utm_easting": args.utm_easting,
        "utm_northing": args.utm_northing,
    }

    # Write GeoTIFF
    output_path = os.path.abspath(args.output)
    write_geotiff(elev, transform, crs, nodata, output_path)
    print(f"Wrote: {output_path}")

    # Print summary
    print_summary(params, expectations)

    # Spot-check corners
    print("\nSpot checks:")
    channel_col = expectations["channel_col"]
    print(f"  elev[0, 0] (NW corner):     {elev[0, 0]:.4f} m")
    print(f"  elev[0, {channel_col}] (N channel): {elev[0, channel_col]:.4f} m")
    print(
        f"  elev[{args.nrows - 1}, {channel_col}] (S channel): {elev[args.nrows - 1, channel_col]:.4f} m"
    )
    print(f"  elev[{args.nrows - 1}, 0] (SW corner): {elev[args.nrows - 1, 0]:.4f} m")
    print(
        f"  elev[{args.nrows - 1}, {args.ncols - 1}] (SE corner): {elev[args.nrows - 1, args.ncols - 1]:.4f} m"
    )


if __name__ == "__main__":
    main()
