"""Tests for split-valley DEM: flow-path DTND vs EDT DTND.

The split valley has two V-valleys side by side. Channel B sits 3.01m higher
than Channel A, pushing the drainage divide ~10 columns past the geometric
midpoint. This creates a "divergence zone" where pixels are geographically
closer to Channel B but hydrologically drain to Channel A.

The V-valley cannot detect an EDT bug because every pixel's nearest stream IS
its flow-path target. This DEM breaks that equivalence.
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

# Import split-valley generator
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data.synthetic_valley.generate_split_valley import generate_split_valley  # noqa: E402

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# Split-valley parameters (from the generator)
NROWS = 200
NCOLS = 1000
PIXEL_SIZE = 5.0
CROSS_SLOPE = 0.03
DOWNSTREAM_SLOPE = 0.001
BASE_ELEVATION = 50.0
CHANNEL_OFFSET = 3.01
CHANNEL_A_COL = 200
CHANNEL_B_COL = 700
EPSG = 32617
UTM_EASTING = 400000.0
UTM_NORTHING = 3286000.0
NODATA = -9999.0


def _channel_threshold(divide_col_int):
    """Compute accumulation threshold that selects only channel columns.

    With purely E-W D8 routing, non-channel pixels accumulate lateral flow
    from one side only. The maximum lateral accumulation is the longest
    distance from a channel to an edge or divide:
      max(CHANNEL_A_COL, divide - CHANNEL_A_COL,
          CHANNEL_B_COL - divide - 1, NCOLS - 1 - CHANNEL_B_COL) ≈ 300

    Each channel pixel accumulates flow from its entire basin width (460+)
    at every row. The threshold must be between these two values.
    """
    max_lateral = max(
        CHANNEL_A_COL,
        divide_col_int - CHANNEL_A_COL,
        CHANNEL_B_COL - divide_col_int - 1,
        NCOLS - 1 - CHANNEL_B_COL,
    )
    min_basin_width = min(divide_col_int + 1, NCOLS - divide_col_int - 1)
    return (max_lateral + min_basin_width) // 2


@pytest.fixture(scope="module")
def split_valley_path():
    """Generate split-valley DEM and write to a temporary GeoTIFF."""
    elev, _ = generate_split_valley()

    transform = Affine(PIXEL_SIZE, 0, UTM_EASTING, 0, -PIXEL_SIZE, UTM_NORTHING)
    crs = CRS.from_epsg(EPSG)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "split_valley_utm.tif")

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
    """Analytical expectations from the split-valley generator."""
    _, exp = generate_split_valley()
    return exp


@pytest.fixture(scope="module")
def grid_with_dem(split_valley_path):
    """Load split-valley DEM into a Grid."""
    grid = Grid.from_raster(split_valley_path, "dem")
    return grid


@pytest.fixture(scope="module")
def processed_grid(grid_with_dem):
    """Run flow routing on the split-valley DEM.

    No flats or depressions exist in the split valley, so resolve_flats
    is skipped — flowdir is computed directly on the raw DEM.
    """
    grid = grid_with_dem
    grid.flowdir("dem", out_name="fdir", dirmap=DIRMAP, routing="d8")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")
    return grid


@pytest.fixture(scope="module")
def grid_with_hand(processed_grid, expectations):
    """Compute HAND, DTND on the split-valley DEM.

    Channel threshold is derived from the geometry to select only
    the two channel columns. See _channel_threshold() for details.
    """
    grid = processed_grid
    threshold = _channel_threshold(expectations["divide_col_int"])
    acc_mask = grid.acc > threshold
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


class TestFlowRouting:
    """Verify D8 routing assigns pixels to the correct basin."""

    def test_flow_directions_are_cardinal(self, processed_grid):
        """Interior pixels should have E or W flow directions.

        cross_slope >> downstream_slope ensures lateral routing dominates.
        DIRMAP: E=1, W=16.
        """
        grid = processed_grid
        fdir = grid.fdir
        edge = 5

        # Sample a band of interior pixels in basin A (away from channels/edges)
        interior_fdir = fdir[edge:-edge, 50:180]
        unique_dirs = set(np.unique(interior_fdir))
        cardinal_ew = {1, 16}  # E, W in DIRMAP
        assert unique_dirs.issubset(cardinal_ew), (
            f"Expected only E(1)/W(16) directions, got {unique_dirs}"
        )

    def test_divide_location(self, processed_grid, expectations):
        """Drainage divide should be at the analytically predicted column.

        Col divide_col_int is the last column draining to A (routes west).
        Col divide_col_int + 1 is the first column draining to B (routes east).
        DIRMAP: W=16, E=1.
        """
        fdir = processed_grid.fdir
        div = expectations["divide_col_int"]

        # Check at 3 rows spread across the domain
        test_rows = [NROWS // 4, NROWS // 2, 3 * NROWS // 4]
        for r in test_rows:
            assert fdir[r, div] == 16, (
                f"Row {r}, col {div} (divide): expected W(16), got {fdir[r, div]}"
            )
            assert fdir[r, div + 1] == 1, (
                f"Row {r}, col {div + 1} (past divide): expected E(1), "
                f"got {fdir[r, div + 1]}"
            )

    def test_channel_a_has_high_accumulation(self, processed_grid, expectations):
        """Channel A accumulation at the outlet should equal its basin area."""
        grid = processed_grid
        divide = expectations["divide_col_int"]

        # Last row: channel A collects flow from all rows
        # Expected basin A area ≈ (divide + 1) * NROWS
        acc_a = grid.acc[NROWS - 1, CHANNEL_A_COL]
        expected_min = (divide + 1) * NROWS * 0.9
        assert acc_a >= expected_min, (
            f"Channel A outlet acc = {acc_a}, expected >= {expected_min:.0f}"
        )

    def test_channel_b_has_high_accumulation(self, processed_grid, expectations):
        """Channel B accumulation at the outlet should equal its basin area."""
        grid = processed_grid
        divide = expectations["divide_col_int"]

        acc_b = grid.acc[NROWS - 1, CHANNEL_B_COL]
        expected_min = (NCOLS - divide - 1) * NROWS * 0.9
        assert acc_b >= expected_min, (
            f"Channel B outlet acc = {acc_b}, expected >= {expected_min:.0f}"
        )


class TestHAND:
    """Test HAND computation on the split-valley DEM."""

    def test_hand_non_negative(self, grid_with_hand):
        """HAND should be non-negative everywhere."""
        hand = grid_with_hand.hand
        valid = hand[~np.isnan(hand)]
        assert np.all(valid >= 0), f"Negative HAND values: min={valid.min():.4f}"

    def test_hand_channel_zero(self, grid_with_hand):
        """HAND at both channel columns should be zero."""
        hand = grid_with_hand.hand
        edge = 5

        for col, name in [(CHANNEL_A_COL, "A"), (CHANNEL_B_COL, "B")]:
            channel_hand = hand[edge:-edge, col]
            assert np.all(channel_hand == 0), (
                f"Channel {name} HAND should be 0, got range "
                f"[{channel_hand.min():.4f}, {channel_hand.max():.4f}]"
            )

    def test_hand_basin_a(self, grid_with_hand, expectations):
        """Basin A interior: HAND should follow analytical formula."""
        hand = grid_with_hand.hand
        edge = 5
        margin = 5
        div_end = expectations["divide_col_int"]

        cols = np.arange(NCOLS)
        expected_hand_row = CROSS_SLOPE * np.abs(cols - CHANNEL_A_COL) * PIXEL_SIZE
        tol = CROSS_SLOPE * PIXEL_SIZE

        for r in range(edge, NROWS - edge, 20):
            actual = hand[r, CHANNEL_A_COL + margin + 1 : div_end - margin]
            expected = expected_hand_row[CHANNEL_A_COL + margin + 1 : div_end - margin]
            np.testing.assert_allclose(
                actual,
                expected,
                atol=tol,
                err_msg=f"HAND mismatch in basin A, row {r}",
            )

    def test_hand_basin_b(self, grid_with_hand, expectations):
        """Basin B interior: HAND should follow analytical formula."""
        hand = grid_with_hand.hand
        edge = 5
        margin = 5
        div_end = expectations["divide_col_int"]

        cols = np.arange(NCOLS)
        expected_hand_row = CROSS_SLOPE * np.abs(cols - CHANNEL_B_COL) * PIXEL_SIZE
        tol = CROSS_SLOPE * PIXEL_SIZE

        for r in range(edge, NROWS - edge, 20):
            actual = hand[r, div_end + 1 + margin : CHANNEL_B_COL - margin]
            expected = expected_hand_row[div_end + 1 + margin : CHANNEL_B_COL - margin]
            np.testing.assert_allclose(
                actual,
                expected,
                atol=tol,
                err_msg=f"HAND mismatch in basin B, row {r}",
            )

    def test_hand_divergence_zone(self, grid_with_hand, expectations):
        """Divergence zone: HAND follows channel A distances, not channel B.

        Checks multiple rows to ensure the divergence zone behavior is
        consistent across the domain, not a single-row coincidence.
        """
        hand = grid_with_hand.hand
        div_end = expectations["divide_col_int"]
        edge = 5

        expected_hand_a = CROSS_SLOPE * abs(div_end - CHANNEL_A_COL) * PIXEL_SIZE
        wrong_hand_b = CROSS_SLOPE * abs(div_end - CHANNEL_B_COL) * PIXEL_SIZE
        tol = CROSS_SLOPE * PIXEL_SIZE

        test_rows = [NROWS // 4, NROWS // 3, NROWS // 2, 2 * NROWS // 3, 3 * NROWS // 4]
        test_rows = [r for r in test_rows if edge <= r < NROWS - edge]

        for r in test_rows:
            actual_hand = hand[r, div_end]
            np.testing.assert_allclose(
                actual_hand,
                expected_hand_a,
                atol=tol,
                err_msg=(
                    f"Row {r}, divergence zone HAND at col {div_end}: "
                    f"got {actual_hand:.2f}, expected {expected_hand_a:.2f} (via A), "
                    f"not {wrong_hand_b:.2f} (via B)"
                ),
            )
            assert abs(actual_hand - expected_hand_a) < abs(
                actual_hand - wrong_hand_b
            ), (
                f"Row {r}: HAND in divergence zone is closer to channel B "
                f"value than channel A"
            )


class TestDTND:
    """Test DTND computation — the key test for this DEM."""

    def test_dtnd_non_negative(self, grid_with_hand):
        """DTND should be non-negative everywhere."""
        dtnd = grid_with_hand.dtnd
        valid = dtnd[~np.isnan(dtnd)]
        assert np.all(valid >= 0), f"Negative DTND values: min={valid.min():.4f}"

    def test_dtnd_channel_zero(self, grid_with_hand):
        """DTND at both channel columns should be zero."""
        dtnd = grid_with_hand.dtnd
        edge = 5

        for col, name in [(CHANNEL_A_COL, "A"), (CHANNEL_B_COL, "B")]:
            channel_dtnd = dtnd[edge:-edge, col]
            assert np.all(channel_dtnd == 0), (
                f"Channel {name} DTND should be 0, got range "
                f"[{channel_dtnd.min():.2f}, {channel_dtnd.max():.2f}]"
            )

    def test_dtnd_basin_a(self, grid_with_hand, expectations):
        """Basin A interior: DTND = |c - channel_A| * pixel_size."""
        dtnd = grid_with_hand.dtnd
        edge = 5
        margin = 5
        div_end = expectations["divide_col_int"]

        cols = np.arange(NCOLS)
        expected_dtnd_row = np.abs(cols - CHANNEL_A_COL) * PIXEL_SIZE

        for r in range(edge, NROWS - edge, 20):
            actual = dtnd[r, CHANNEL_A_COL + margin + 1 : div_end - margin]
            expected = expected_dtnd_row[CHANNEL_A_COL + margin + 1 : div_end - margin]
            np.testing.assert_allclose(
                actual,
                expected,
                atol=PIXEL_SIZE,
                err_msg=f"DTND mismatch in basin A, row {r}",
            )

    def test_dtnd_basin_b(self, grid_with_hand, expectations):
        """Basin B interior: DTND = |c - channel_B| * pixel_size."""
        dtnd = grid_with_hand.dtnd
        edge = 5
        margin = 5
        div_end = expectations["divide_col_int"]

        cols = np.arange(NCOLS)
        expected_dtnd_row = np.abs(cols - CHANNEL_B_COL) * PIXEL_SIZE

        for r in range(edge, NROWS - edge, 20):
            actual = dtnd[r, div_end + 1 + margin : CHANNEL_B_COL - margin]
            expected = expected_dtnd_row[div_end + 1 + margin : CHANNEL_B_COL - margin]
            np.testing.assert_allclose(
                actual,
                expected,
                atol=PIXEL_SIZE,
                err_msg=f"DTND mismatch in basin B, row {r}",
            )

    def test_dtnd_divergence_zone(self, grid_with_hand, expectations):
        """Core test: divergence zone DTND follows flow paths, not EDT.

        For each sample column, actual DTND should match the flow-path
        distance to channel A (farther away), not the EDT distance to
        channel B (geographically closer).
        """
        dtnd = grid_with_hand.dtnd
        mid_row = NROWS // 2
        div_samples = expectations["divergence_samples"]

        for col, vals in sorted(div_samples.items()):
            actual = dtnd[mid_row, col]
            flow_dtnd = vals["flow_dtnd"]
            edt_dtnd = vals["edt_dtnd"]

            np.testing.assert_allclose(
                actual,
                flow_dtnd,
                atol=PIXEL_SIZE,
                err_msg=(
                    f"Col {col}: DTND={actual:.1f}m, "
                    f"expected flow_dtnd={flow_dtnd:.1f}m, "
                    f"not edt_dtnd={edt_dtnd:.1f}m"
                ),
            )
            assert abs(actual - flow_dtnd) < abs(actual - edt_dtnd), (
                f"Col {col}: DTND ({actual:.1f}m) is closer to EDT ({edt_dtnd:.1f}m) "
                f"than flow-path ({flow_dtnd:.1f}m)"
            )

    def test_dtnd_not_haversine_garbage(self, grid_with_hand):
        """Max DTND should be bounded by domain width, not haversine garbage."""
        dtnd = grid_with_hand.dtnd
        domain_width = NCOLS * PIXEL_SIZE  # 5000m

        valid = dtnd[~np.isnan(dtnd)]
        assert np.max(valid) < domain_width, (
            f"Max DTND = {np.max(valid):.1f}m, domain width = {domain_width:.0f}m. "
            f"Likely haversine-on-UTM bug."
        )


class TestHandDtndRelationship:
    """Cross-validate HAND and DTND using the split valley's analytical geometry."""

    def test_hand_equals_cross_slope_times_dtnd(self, grid_with_hand, expectations):
        """On interior pixels of both basins: HAND = cross_slope * DTND."""
        hand = grid_with_hand.hand
        dtnd = grid_with_hand.dtnd
        edge = 5
        margin = 5
        div_end = expectations["divide_col_int"]
        tol = CROSS_SLOPE * PIXEL_SIZE

        # Basin A interior
        a_hand = hand[edge:-edge, CHANNEL_A_COL + margin + 1 : div_end - margin]
        a_dtnd = dtnd[edge:-edge, CHANNEL_A_COL + margin + 1 : div_end - margin]
        np.testing.assert_allclose(
            a_hand,
            CROSS_SLOPE * a_dtnd,
            atol=tol,
            err_msg="HAND != cross_slope * DTND in basin A",
        )

        # Basin B interior
        b_hand = hand[edge:-edge, div_end + 1 + margin : CHANNEL_B_COL - margin]
        b_dtnd = dtnd[edge:-edge, div_end + 1 + margin : CHANNEL_B_COL - margin]
        np.testing.assert_allclose(
            b_hand,
            CROSS_SLOPE * b_dtnd,
            atol=tol,
            err_msg="HAND != cross_slope * DTND in basin B",
        )
