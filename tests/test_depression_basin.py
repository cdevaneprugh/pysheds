"""Tests for depression basin DEM: fill_depressions and resolve_flats.

The V-valley and split valley have no flats or depressions — they skip the
DEM conditioning chain entirely. This DEM exercises the full pipeline:
fill_depressions -> resolve_flats -> flowdir -> accumulation -> HAND/DTND.

The depression is a parabolic bowl on a north-to-south tilted plane. After
filling, the bowl becomes a flat at the spill elevation. After resolve_flats,
micro-gradients route flow across the flat toward the spill point.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from pysheds.pgrid import Grid

# Import depression basin generator
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data.synthetic_valley.generate_depression_basin import (  # noqa: E402
    generate_depression_basin,
)

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
# DIRMAP encoding: N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32
VALID_FDIR_VALUES = {1, 2, 4, 8, 16, 32, 64, 128}

# Depression basin parameters (from the generator)
NROWS = 200
NCOLS = 200
PIXEL_SIZE = 5.0
BACKGROUND_SLOPE = 0.01
BASE_ELEVATION = 50.0
DEP_CENTER_ROW = 100
DEP_CENTER_COL = 100
DEP_RADIUS_PX = 30
DEP_DEPTH = 3.0
EPSG = 32617
UTM_EASTING = 400000.0
UTM_NORTHING = 3286000.0
NODATA = -9999.0


@pytest.fixture(scope="module")
def depression_basin_path():
    """Generate depression basin DEM and write to a temporary GeoTIFF."""
    elev, _ = generate_depression_basin()

    transform = Affine(PIXEL_SIZE, 0, UTM_EASTING, 0, -PIXEL_SIZE, UTM_NORTHING)
    crs = CRS.from_epsg(EPSG)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "depression_basin_utm.tif")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=NROWS,
        width=NCOLS,
        count=1,
        dtype=elev.dtype,
        crs=crs,
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(elev, 1)

    yield path

    os.unlink(path)
    os.rmdir(tmpdir)


@pytest.fixture(scope="module")
def expectations():
    """Analytical expectations from the depression basin generator."""
    _, exp = generate_depression_basin()
    return exp


@pytest.fixture(scope="module")
def raw_grid(depression_basin_path):
    """Load unfilled DEM into a Grid."""
    grid = Grid.from_raster(depression_basin_path, "dem")
    return grid


@pytest.fixture(scope="module")
def filled_grid(raw_grid):
    """Run fill_depressions on the raw DEM."""
    grid = raw_grid
    grid.fill_depressions("dem", out_name="flooded_dem")
    return grid


@pytest.fixture(scope="module")
def conditioned_grid(filled_grid):
    """Run resolve_flats on the filled DEM."""
    grid = filled_grid
    grid.resolve_flats("flooded_dem", out_name="inflated_dem")
    return grid


@pytest.fixture(scope="module")
def routed_grid(conditioned_grid):
    """Run flowdir and accumulation on the conditioned DEM."""
    grid = conditioned_grid
    grid.flowdir("inflated_dem", out_name="fdir", dirmap=DIRMAP, routing="d8")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")
    return grid


@pytest.fixture(scope="module")
def grid_with_hand(routed_grid):
    """Compute HAND and DTND after full conditioning chain.

    Channel threshold: acc > NROWS - 1. On the uniform background slope,
    accumulation at (r, c) = r + 1 (purely southward flow — each pixel
    gets flow only from pixels directly north). South edge pixels (row
    NROWS-1) have acc = NROWS. With acc > NROWS - 1, the south edge row
    and the depression outflow column qualify as channel, so every pixel
    on the background slope can trace south to a channel pixel.
    """
    grid = routed_grid
    acc_mask = grid.acc > NROWS - 1
    grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)
    grid.compute_hand(
        "fdir",
        "dem",
        grid.channel_mask,
        grid.channel_id,
        dirmap=DIRMAP,
        routing="d8",
    )
    return grid


def _depression_mask():
    """Reconstruct the depression mask from constants (independent of generator)."""
    rr, cc = np.meshgrid(
        np.arange(NROWS, dtype=np.float64),
        np.arange(NCOLS, dtype=np.float64),
        indexing="ij",
    )
    dist_px = np.sqrt((rr - DEP_CENTER_ROW) ** 2 + (cc - DEP_CENTER_COL) ** 2)
    return dist_px < DEP_RADIUS_PX, dist_px


class TestFillDepressions:
    """Verify depression filling correctness."""

    def test_fill_raises_to_spill_elevation(self, filled_grid, expectations):
        """All depression pixels below spill should be raised to spill level."""
        flooded = filled_grid.flooded_dem
        dem = filled_grid.dem
        spill_elev = expectations["spill_elev"]
        inside, _ = _depression_mask()

        was_below = inside & (dem < spill_elev)
        if np.any(was_below):
            filled_values = flooded[was_below]
            assert np.all(filled_values >= spill_elev - 1e-4), (
                f"Some filled pixels are below spill elevation {spill_elev:.4f}: "
                f"min={filled_values.min():.4f}"
            )

    def test_fill_does_not_lower(self, filled_grid):
        """fill_depressions should only raise elevations, never lower them."""
        flooded = filled_grid.flooded_dem
        dem = filled_grid.dem
        assert np.all(flooded >= dem - 1e-6), (
            f"fill_depressions lowered some pixels: "
            f"min(flooded - dem) = {(flooded - dem).min():.6f}"
        )

    def test_fill_preserves_outside(self, filled_grid):
        """Pixels outside the depression should be unchanged."""
        flooded = filled_grid.flooded_dem
        dem = filled_grid.dem
        _, dist_px = _depression_mask()

        outside = dist_px >= DEP_RADIUS_PX + 2
        np.testing.assert_array_equal(
            flooded[outside],
            dem[outside],
            err_msg="fill_depressions modified pixels outside depression",
        )

    def test_filled_region_is_flat(self, filled_grid, expectations):
        """Pixels that were below spill should now be at spill elevation."""
        flooded = filled_grid.flooded_dem
        dem = filled_grid.dem
        spill_elev = expectations["spill_elev"]
        inside, _ = _depression_mask()

        was_below = inside & (dem < spill_elev)
        if np.any(was_below):
            filled_values = flooded[was_below]
            np.testing.assert_allclose(
                filled_values,
                spill_elev,
                atol=1e-4,
                err_msg="Filled region is not uniformly at spill elevation",
            )


class TestResolveFlats:
    """Verify flat resolution via Garbrecht-Martz algorithm."""

    def test_no_interior_flats_remain(self, conditioned_grid, expectations):
        """In the filled depression region, no pixel should have all 8
        neighbors at the exact same elevation in the inflated DEM.
        """
        inflated = conditioned_grid.inflated_dem
        spill_elev = expectations["spill_elev"]
        dem = conditioned_grid.dem
        inside, _ = _depression_mask()

        was_flat = inside & (dem < spill_elev)
        flat_rows, flat_cols = np.where(was_flat)

        n_checked = 0
        n_still_flat = 0
        for r, c in zip(flat_rows[::10], flat_cols[::10]):
            if r <= 0 or r >= NROWS - 1 or c <= 0 or c >= NCOLS - 1:
                continue
            center = inflated[r, c]
            neighbors = inflated[r - 1 : r + 2, c - 1 : c + 2].ravel()
            neighbor_vals = np.concatenate([neighbors[:4], neighbors[5:]])
            if np.all(neighbor_vals == center):
                n_still_flat += 1
            n_checked += 1

        assert n_checked > 0, "No interior flat pixels found to check"
        assert n_still_flat == 0, (
            f"{n_still_flat}/{n_checked} sampled pixels still have all 8 "
            f"neighbors at equal elevation after resolve_flats"
        )

    def test_inflated_does_not_lower_filled(self, conditioned_grid):
        """resolve_flats should only add micro-gradients upward."""
        inflated = conditioned_grid.inflated_dem
        flooded = conditioned_grid.flooded_dem

        assert np.all(inflated >= flooded - 1e-6), (
            f"resolve_flats lowered some pixels: "
            f"min(inflated - flooded) = {(inflated - flooded).min():.6f}"
        )


class TestFlowDirection:
    """D8 routing after the full conditioning chain."""

    def test_interior_pixels_have_valid_direction(self, routed_grid):
        """Every interior pixel should have a valid D8 flow direction.

        Edge pixels (row 0, row N-1, col 0, col N-1) may have fdir=0
        as a boundary condition, so we only check interior pixels.
        """
        fdir = routed_grid.fdir
        interior = fdir[1:-1, 1:-1]
        unique_dirs = set(np.unique(interior))
        invalid = unique_dirs - VALID_FDIR_VALUES
        assert not invalid, (
            f"Invalid flow direction values in interior: {invalid}. "
            f"All values: {unique_dirs}"
        )

    def test_background_flows_south(self, routed_grid):
        """Outside the depression, the background slope (N-to-S) should
        produce southward flow directions. S=4 in DIRMAP.
        """
        fdir = routed_grid.fdir

        # Sample a column far from the depression (col 10, rows 5-50)
        far_col_fdir = fdir[5:50, 10]
        assert np.all(far_col_fdir == 4), (
            f"Background column should flow south (4 in DIRMAP), "
            f"got unique values: {np.unique(far_col_fdir)}"
        )

    def test_no_pits_remain(self, conditioned_grid):
        """No interior pixel should be a local minimum in the inflated DEM."""
        inflated = np.asarray(conditioned_grid.inflated_dem, dtype=np.float64)

        # Vectorized: compare center to all 8 neighbors
        center = inflated[1:-1, 1:-1]
        is_pit = np.ones_like(center, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = inflated[1 + dr : NROWS - 1 + dr, 1 + dc : NCOLS - 1 + dc]
                is_pit &= center < neighbor

        n_pits = np.sum(is_pit)
        assert n_pits == 0, f"{n_pits} interior pits remain after conditioning"


class TestAccumulation:
    """Flow accumulation after the full conditioning chain."""

    def test_depression_outflow_accumulation(self, routed_grid, expectations):
        """The depression outflow should produce a high-accumulation column
        at the south edge that far exceeds background-only accumulation.

        On the pure background slope, each column at the south edge has
        acc ≈ NROWS (purely southward flow). The depression adds its entire
        area (~2800 pixels) to the outflow column, so the max accumulation
        at the south edge should exceed NROWS + n_depression_px.
        """
        acc = routed_grid.acc
        n_depression = expectations["n_depression_px"]

        max_acc_bottom = np.max(acc[NROWS - 1, :])
        assert max_acc_bottom > NROWS + n_depression, (
            f"Max accumulation at south edge = {max_acc_bottom}, "
            f"expected > {NROWS + n_depression} "
            f"(NROWS={NROWS} + depression area={n_depression})"
        )

    def test_max_accumulation_at_south_edge(self, routed_grid):
        """Maximum accumulation should be in the bottom row (outlet)."""
        acc = routed_grid.acc
        max_row = np.unravel_index(np.argmax(acc), acc.shape)[0]
        assert max_row == NROWS - 1, (
            f"Max accumulation at row {max_row}, expected row {NROWS - 1} (south edge)"
        )


class TestHAND:
    """HAND after the full conditioning chain."""

    def test_hand_bounded_below(self, grid_with_hand):
        """HAND should not be more negative than the depression depth.

        compute_hand() uses the original (unfilled) DEM for elevation
        differences. Pixels inside the depression have original elevation
        below their drainage point, so HAND can be legitimately negative
        (up to DEP_DEPTH below zero).
        """
        hand = grid_with_hand.hand
        valid = hand[~np.isnan(hand)]
        assert np.all(valid >= -DEP_DEPTH - 0.1), (
            f"HAND too negative: min={valid.min():.4f}, "
            f"expected >= {-DEP_DEPTH - 0.1:.1f}"
        )

    def test_hand_channel_zero(self, grid_with_hand):
        """Channel pixels (identified by acc threshold) should have HAND == 0."""
        hand = grid_with_hand.hand
        channel_mask = np.asarray(grid_with_hand.channel_mask).astype(bool)
        channel_hand = hand[channel_mask]
        valid = channel_hand[~np.isnan(channel_hand)]
        assert len(valid) > 0, "No valid channel HAND values — check channel threshold"
        assert np.all(valid == 0), (
            f"Channel pixel HAND should be 0, "
            f"got range [{valid.min():.4f}, {valid.max():.4f}]"
        )

    def test_hand_bounded(self, grid_with_hand):
        """Max HAND should not exceed the domain's total elevation range."""
        hand = grid_with_hand.hand
        max_elev_range = NROWS * PIXEL_SIZE * BACKGROUND_SLOPE + DEP_DEPTH
        valid = hand[~np.isnan(hand)]
        assert len(valid) > 0, "No valid HAND values — check channel threshold"
        assert np.max(valid) < max_elev_range, (
            f"Max HAND = {np.max(valid):.2f}m, "
            f"exceeds domain elevation range {max_elev_range:.2f}m"
        )


class TestDTND:
    """DTND after the full conditioning chain."""

    def test_dtnd_non_negative(self, grid_with_hand):
        """DTND should be non-negative everywhere."""
        dtnd = grid_with_hand.dtnd
        valid = dtnd[~np.isnan(dtnd)]
        assert np.all(valid >= 0), f"Negative DTND values: min={valid.min():.4f}"

    def test_dtnd_channel_zero(self, grid_with_hand):
        """Channel pixels should have DTND == 0."""
        dtnd = grid_with_hand.dtnd
        channel_mask = np.asarray(grid_with_hand.channel_mask).astype(bool)
        channel_dtnd = dtnd[channel_mask]
        valid = channel_dtnd[~np.isnan(channel_dtnd)]
        assert len(valid) > 0, "No valid channel DTND values — check channel threshold"
        assert np.all(valid == 0), (
            f"Channel pixel DTND should be 0, "
            f"got range [{valid.min():.2f}, {valid.max():.2f}]"
        )

    def test_dtnd_bounded(self, grid_with_hand):
        """Max DTND should not exceed the domain diagonal."""
        dtnd = grid_with_hand.dtnd
        domain_diagonal = np.sqrt(NROWS**2 + NCOLS**2) * PIXEL_SIZE

        valid = dtnd[~np.isnan(dtnd)]
        assert len(valid) > 0, "No valid DTND values — check channel threshold"
        assert np.max(valid) < domain_diagonal, (
            f"Max DTND = {np.max(valid):.1f}m, "
            f"exceeds domain diagonal {domain_diagonal:.1f}m"
        )

    def test_dtnd_finite_where_hand_valid(self, grid_with_hand):
        """DTND should be finite for all pixels with a valid drainage path.

        On the background slope, every pixel drains south to the channel row.
        This creates dx=0 with dy!=0 in the Euclidean DTND formula.
        If the formula has a sign error (e.g. dx^2 - dy^2 instead of +),
        sqrt of a negative produces NaN — which the bound-checking tests miss
        because they filter NaN before asserting.
        """
        dtnd = grid_with_hand.dtnd
        hand = grid_with_hand.hand

        # Pixels with valid HAND (not NaN) have a drainage path via hndx,
        # so their DTND must also be finite.
        has_valid_hand = ~np.isnan(hand)
        has_nan_dtnd = np.isnan(dtnd)
        invalid = has_valid_hand & has_nan_dtnd
        assert not np.any(invalid), (
            f"{np.sum(invalid)} pixels have valid HAND but NaN DTND — "
            f"Euclidean distance formula may have a sign error"
        )
