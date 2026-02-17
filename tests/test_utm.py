"""Tests for UTM CRS support in pysheds.

Validates that slope_aspect(), compute_hand() DTND, and compute_hand() AZND
produce correct results on projected (UTM) coordinate data. Uses a synthetic
V-valley DEM with analytically known outputs.

The V-valley has two planar hillslopes meeting at a central N-S channel.
Every parameter has a closed-form solution, so deviations indicate CRS bugs.

Key design choice: pixel_size=5m (not 1m) to ensure the code doesn't
accidentally pass by treating coordinates as unit-spaced.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from pysheds.pgrid import Grid

# Import V-valley generator
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data.synthetic_valley.generate_synthetic_dem import generate_v_valley  # noqa: E402

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)

# V-valley parameters: 200x200 grid at 5m pixel size
# This gives a 1km x 1km domain -- large enough for meaningful flow routing,
# small enough for fast tests.
NROWS = 200
NCOLS = 200
PIXEL_SIZE = 5.0
CROSS_SLOPE = 0.03
DOWNSTREAM_SLOPE = 0.001
BASE_ELEVATION = 50.0
UTM_EASTING = 400000.0
UTM_NORTHING = 3286000.0
EPSG = 32617


@pytest.fixture(scope="module")
def v_valley_path():
    """Generate V-valley DEM and write to a temporary GeoTIFF."""
    elev, _ = generate_v_valley(
        nrows=NROWS,
        ncols=NCOLS,
        pixel_size=PIXEL_SIZE,
        cross_slope=CROSS_SLOPE,
        downstream_slope=DOWNSTREAM_SLOPE,
        base_elevation=BASE_ELEVATION,
    )

    transform = Affine(PIXEL_SIZE, 0, UTM_EASTING, 0, -PIXEL_SIZE, UTM_NORTHING)
    crs = CRS.from_epsg(EPSG)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "v_valley_utm.tif")

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
        nodata=-9999.0,
    ) as dst:
        dst.write(elev, 1)

    yield path

    # Cleanup
    os.unlink(path)
    os.rmdir(tmpdir)


@pytest.fixture(scope="module")
def expectations():
    """Analytical expectations for the V-valley DEM."""
    _, exp = generate_v_valley(
        nrows=NROWS,
        ncols=NCOLS,
        pixel_size=PIXEL_SIZE,
        cross_slope=CROSS_SLOPE,
        downstream_slope=DOWNSTREAM_SLOPE,
        base_elevation=BASE_ELEVATION,
    )
    return exp


@pytest.fixture(scope="module")
def grid_with_dem(v_valley_path):
    """Load V-valley DEM into a Grid."""
    grid = Grid.from_raster(v_valley_path, "dem")
    return grid


@pytest.fixture(scope="module")
def processed_grid(grid_with_dem):
    """Run full flow routing pipeline on V-valley DEM.

    The V-valley has no flat regions (every pixel has a unique steepest
    descent neighbor), so resolve_flats is skipped — it crashes on
    the edge case of zero flat cells. flowdir is computed directly.
    """
    grid = grid_with_dem

    grid.flowdir("dem", out_name="fdir", dirmap=DIRMAP, routing="d8")
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    return grid


@pytest.fixture(scope="module")
def grid_with_slope_aspect(grid_with_dem):
    """Compute slope and aspect on the V-valley DEM."""
    grid = grid_with_dem
    grid.slope_aspect("dem")
    return grid


@pytest.fixture(scope="module")
def grid_with_hand(processed_grid):
    """Compute HAND, DTND, AZND on the V-valley DEM.

    Uses acc > threshold to define channels. The center column (col 100)
    accumulates ~19800 cells, while adjacent columns have ~99. A threshold
    of NROWS ensures only the center column is classified as channel.
    """
    grid = processed_grid

    # Only the center channel column has acc >> NROWS (it collects lateral
    # flow from both sides across all rows). Adjacent columns have acc < NCOLS/2.
    acc_mask = grid.acc > NROWS

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


@pytest.fixture(scope="module")
def grid_with_river_stats(processed_grid):
    """Run river_network_length_and_slope on the V-valley UTM DEM.

    The V-valley has no flats, so resolve_flats was skipped; aliasing
    inflated_dem = dem is correct (no inflation needed). The function
    needs inflated_dem to extract channel elevation profiles.
    """
    grid = processed_grid
    grid.inflated_dem = grid.dem
    acc_mask = grid.acc > NROWS
    result = grid.river_network_length_and_slope("fdir", acc_mask, dirmap=DIRMAP)
    return result


class TestCRSDetection:
    """Test that the CRS helper correctly identifies geographic vs projected."""

    def test_utm_is_not_geographic(self, grid_with_dem):
        assert not grid_with_dem._crs_is_geographic()

    def test_geographic_dem_is_geographic(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dem_path = os.path.join(current_dir, "..", "data", "dem.tif")
        grid = Grid.from_raster(dem_path, "dem")
        assert grid._crs_is_geographic()


class TestSlope:
    """Test slope computation on V-valley UTM DEM."""

    def test_slope_magnitude(self, grid_with_slope_aspect, expectations):
        """Slope should match sqrt(cross_slope^2 + downstream_slope^2).

        Exclude a 3-pixel band around the channel where the Horn stencil
        averages across the V-valley gradient discontinuity.
        """
        grid = grid_with_slope_aspect
        expected_slope = expectations["slope"]
        channel_col = expectations["channel_col"]

        slope = grid.slope

        # Interior pixels away from channel and edges
        # Horn stencil uses 3x3 window, so skip 1 pixel from edges and
        # 3 pixels from channel center (stencil spans the discontinuity)
        margin = 3
        edge = 2

        west_slope = slope[edge:-edge, edge : channel_col - margin]
        east_slope = slope[edge:-edge, channel_col + margin + 1 : -edge]

        # Tolerance: 0.001 m/m (well below the 0.03 signal)
        assert west_slope == pytest.approx(expected_slope, abs=0.001)
        assert east_slope == pytest.approx(expected_slope, abs=0.001)


class TestAspect:
    """Test aspect computation on V-valley UTM DEM."""

    def test_aspect_west_side(self, grid_with_slope_aspect, expectations):
        """West-side pixels should face east (~92 deg)."""
        grid = grid_with_slope_aspect
        expected = expectations["aspect_west_side"]
        channel_col = expectations["channel_col"]

        margin = 3
        edge = 2

        west_aspect = grid.aspect[edge:-edge, edge : channel_col - margin]

        # Tolerance: 0.5 deg
        assert west_aspect == pytest.approx(expected, abs=0.5)

    def test_aspect_east_side(self, grid_with_slope_aspect, expectations):
        """East-side pixels should face west (~268 deg)."""
        grid = grid_with_slope_aspect
        expected = expectations["aspect_east_side"]
        channel_col = expectations["channel_col"]

        margin = 3
        edge = 2

        east_aspect = grid.aspect[edge:-edge, channel_col + margin + 1 : -edge]

        # Tolerance: 0.5 deg
        assert east_aspect == pytest.approx(expected, abs=0.5)


class TestHAND:
    """Test HAND computation on V-valley UTM DEM.

    Pipeline validation, not CRS tests. HAND is a pure elevation difference
    (dem[pixel] - dem[drainage_pixel]) — no distance computation involved.
    These verify that D8 routing and HAND computation work on UTM data, not
    that distance calculations use the correct CRS formula.
    """

    def test_hand_values(self, grid_with_hand, expectations):
        """HAND should equal cross_slope * abs(col - channel_col) * pixel_size.

        Skip pixels near channel (stencil effects) and edges (boundary effects).
        """
        grid = grid_with_hand
        cross_slope = expectations["cross_slope"]
        channel_col = expectations["channel_col"]
        pixel_size = expectations["pixel_size"]

        hand = grid.hand

        # Build expected HAND array
        cols = np.arange(NCOLS)
        expected_hand_row = cross_slope * np.abs(cols - channel_col) * pixel_size

        # Check interior pixels away from channel and edges
        margin = 5  # wider margin: channel + boundary effects
        edge = 5

        for r in range(edge, NROWS - edge):
            row_hand = hand[r, :]
            # West side
            west_actual = row_hand[edge : channel_col - margin]
            west_expected = expected_hand_row[edge : channel_col - margin]
            # Tolerance: one pixel width * cross_slope
            tol = cross_slope * pixel_size
            np.testing.assert_allclose(
                west_actual,
                west_expected,
                atol=tol,
                err_msg=f"HAND mismatch on west side, row {r}",
            )
            # East side
            east_actual = row_hand[channel_col + margin + 1 : NCOLS - edge]
            east_expected = expected_hand_row[channel_col + margin + 1 : NCOLS - edge]
            np.testing.assert_allclose(
                east_actual,
                east_expected,
                atol=tol,
                err_msg=f"HAND mismatch on east side, row {r}",
            )


class TestDTND:
    """Test DTND computation on V-valley UTM DEM."""

    def test_dtnd_values(self, grid_with_hand, expectations):
        """DTND should equal abs(col - channel_col) * pixel_size.

        This is the core test: haversine on UTM coordinates produces garbage,
        Euclidean produces the correct answer.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]
        pixel_size = expectations["pixel_size"]

        dtnd = grid.dtnd

        # Build expected DTND array
        cols = np.arange(NCOLS)
        expected_dtnd_row = np.abs(cols - channel_col) * pixel_size

        margin = 5
        edge = 5

        for r in range(edge, NROWS - edge):
            row_dtnd = dtnd[r, :]
            # West side
            west_actual = row_dtnd[edge : channel_col - margin]
            west_expected = expected_dtnd_row[edge : channel_col - margin]
            # Tolerance: one pixel width (flow path may differ by 1 pixel
            # near channel due to DEM conditioning)
            np.testing.assert_allclose(
                west_actual,
                west_expected,
                atol=pixel_size,
                err_msg=f"DTND mismatch on west side, row {r}",
            )
            # East side
            east_actual = row_dtnd[channel_col + margin + 1 : NCOLS - edge]
            east_expected = expected_dtnd_row[channel_col + margin + 1 : NCOLS - edge]
            np.testing.assert_allclose(
                east_actual,
                east_expected,
                atol=pixel_size,
                err_msg=f"DTND mismatch on east side, row {r}",
            )

    def test_dtnd_channel_is_zero(self, grid_with_hand, expectations):
        """DTND at channel pixels should be zero.

        CRS-independent: channel pixels have hndx pointing to themselves
        (pgrid.py:1935), so dlon=dlat=0 -> distance=0 for both haversine
        and Euclidean.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]

        dtnd = grid.dtnd
        edge = 5

        # Channel column should have DTND = 0 (they ARE the drainage)
        channel_dtnd = dtnd[edge:-edge, channel_col]
        assert np.all(channel_dtnd == 0), (
            f"Channel DTND should be 0, got range "
            f"[{channel_dtnd.min():.2f}, {channel_dtnd.max():.2f}]"
        )


class TestAZND:
    """Test AZND (Azimuth to Nearest Drainage) on V-valley UTM DEM.

    The V-valley cross_slope (0.03) >> downstream_slope (0.001), so D8
    routes every hillslope pixel directly east or west to the channel.
    The drainage pixel for a west-side pixel at (r, c) is (r, channel_col),
    giving a pure-east bearing of 90 deg. Similarly, east-side pixels get
    270 deg.
    """

    def test_aznd_west_side(self, grid_with_hand, expectations):
        """West-side pixels drain east to the channel -> AZND ~ 90 deg."""
        grid = grid_with_hand
        channel_col = expectations["channel_col"]

        aznd = grid.aznd
        margin = 5
        edge = 5

        west_aznd = aznd[edge:-edge, edge : channel_col - margin]
        assert west_aznd == pytest.approx(90.0, abs=1.0), (
            f"West-side AZND mean={np.nanmean(west_aznd):.2f}, expected ~90 deg"
        )

    def test_aznd_east_side(self, grid_with_hand, expectations):
        """East-side pixels drain west to the channel -> AZND ~ 270 deg."""
        grid = grid_with_hand
        channel_col = expectations["channel_col"]

        aznd = grid.aznd
        margin = 5
        edge = 5

        east_aznd = aznd[edge:-edge, channel_col + margin + 1 : -edge]
        assert east_aznd == pytest.approx(270.0, abs=1.0), (
            f"East-side AZND mean={np.nanmean(east_aznd):.2f}, expected ~270 deg"
        )

    def test_aznd_channel_points_downstream(self, grid_with_hand, expectations):
        """Channel pixels drain to a downstream channel pixel -> AZND ~ 180 deg.

        CRS-independent: channel hndx points to self, so dlon=dlat=0.
        AZND = arctan2(-0.0, -0.0) = -pi -> 180 deg (IEEE 754 signed-zero
        behavior). Tests hndx topology, not CRS distance computation.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]

        aznd = grid.aznd
        edge = 5

        channel_aznd = aznd[edge:-edge, channel_col]
        assert channel_aznd == pytest.approx(180.0, abs=1.0), (
            f"Channel AZND should be ~180 (south/downstream), got range "
            f"[{channel_aznd.min():.2f}, {channel_aznd.max():.2f}]"
        )


class TestRiverNetworkLengthSlope:
    """Test river_network_length_and_slope() on V-valley UTM DEM.

    CRS test: haversine interprets 5m pixel spacing as 5 degrees of
    latitude, producing ~556 km per segment (~110,000 km total) vs the
    correct ~995m. This test catches that failure mode.
    """

    def test_channel_length(self, grid_with_river_stats):
        """Total reach length should be ~995m, not haversine garbage."""
        result = grid_with_river_stats
        expected_length = (NROWS - 1) * PIXEL_SIZE  # 995m
        assert result["length"] == pytest.approx(expected_length, rel=0.1)

    def test_channel_slope(self, grid_with_river_stats):
        """Mean reach slope should be ~0.001 m/m."""
        result = grid_with_river_stats
        assert result["slope"] == pytest.approx(DOWNSTREAM_SLOPE, rel=0.1)


class TestHandDtndRelationship:
    """Cross-validate HAND and DTND using the V-valley's analytical geometry.

    For the V-valley, every hillslope pixel drains straight E/W to the channel
    column. The drainage pixel for pixel (r, c) is (r, channel_col), so:

        HAND = cross_slope * |c - channel_col| * pixel_size
        DTND = |c - channel_col| * pixel_size

    Therefore: HAND = cross_slope * DTND for all hillslope pixels. This cross-
    validates that HAND and DTND reference the same drainage pixel mapping
    (hndx) and that the Euclidean distance formula is consistent with the
    elevation-difference HAND values.
    """

    def test_hand_equals_cross_slope_times_dtnd(self, grid_with_hand, expectations):
        """HAND / DTND should equal cross_slope on interior hillslope pixels."""
        grid = grid_with_hand
        cross_slope = expectations["cross_slope"]
        channel_col = expectations["channel_col"]

        hand = grid.hand
        dtnd = grid.dtnd

        margin = 5
        edge = 5

        # West side interior pixels
        w_hand = hand[edge:-edge, edge : channel_col - margin]
        w_dtnd = dtnd[edge:-edge, edge : channel_col - margin]

        # East side interior pixels
        e_hand = hand[edge:-edge, channel_col + margin + 1 : -edge]
        e_dtnd = dtnd[edge:-edge, channel_col + margin + 1 : -edge]

        # HAND should equal cross_slope * DTND
        # Tolerance: cross_slope * pixel_size (one pixel worth of error)
        tol = cross_slope * PIXEL_SIZE
        np.testing.assert_allclose(
            w_hand,
            cross_slope * w_dtnd,
            atol=tol,
            err_msg="HAND != cross_slope * DTND on west side",
        )
        np.testing.assert_allclose(
            e_hand,
            cross_slope * e_dtnd,
            atol=tol,
            err_msg="HAND != cross_slope * DTND on east side",
        )


class TestHillslopeClassification:
    """Test compute_hillslope bank classification on V-valley UTM DEM.

    The V-valley has a clean geometry: west side = one bank, east side = the
    other, center = channel. Although compute_hillslope is purely topological
    (no CRS math), exercising it on UTM data provides coverage and validates
    bank separation.
    """

    def test_hillslope_types_present(self, grid_with_hand):
        """Hillslope classification should produce all 4 types."""
        grid = grid_with_hand
        grid.compute_hillslope("fdir", grid.channel_mask, grid.bank_mask, dirmap=DIRMAP)
        hillslope = grid.hillslope
        # Types: 1=headwater, 2=right_bank, 3=left_bank, 4=channel
        present = set(np.unique(hillslope[hillslope > 0]))
        assert present == {1, 2, 3, 4}, f"Expected types {{1,2,3,4}}, got {present}"

    def test_bank_separation(self, grid_with_hand, expectations):
        """West and east sides should have consistent but different bank types."""
        grid = grid_with_hand
        channel_col = expectations["channel_col"]

        # compute_hillslope was already called in test above (same fixture scope)
        hillslope = grid.hillslope

        margin = 5
        edge = 10  # wider edge to avoid headwater classification

        west_types = hillslope[edge:-edge, edge : channel_col - margin]
        east_types = hillslope[edge:-edge, channel_col + margin + 1 : -edge]

        # Each side should be dominated by a single bank type (2 or 3)
        west_mode = np.bincount(west_types.ravel()).argmax()
        east_mode = np.bincount(east_types.ravel()).argmax()

        assert west_mode in {2, 3}, f"West side mode={west_mode}, expected 2 or 3"
        assert east_mode in {2, 3}, f"East side mode={east_mode}, expected 2 or 3"
        assert west_mode != east_mode, (
            f"West and east sides have same bank type ({west_mode})"
        )

    def test_channel_column_is_type_4(self, grid_with_hand, expectations):
        """Channel column pixels should be classified as type 4."""
        grid = grid_with_hand
        channel_col = expectations["channel_col"]
        hillslope = grid.hillslope
        edge = 5

        channel_types = hillslope[edge:-edge, channel_col]
        assert np.all(channel_types == 4), (
            f"Channel column should be type 4, got unique types: "
            f"{np.unique(channel_types)}"
        )


class TestEndToEndUTM:
    """End-to-end integration test: fresh Grid through full pipeline.

    Chains from_raster -> flowdir -> accumulation -> create_channel_mask ->
    compute_hand -> slope_aspect -> compute_hillslope on a fresh Grid to
    catch state-passing bugs between stages that individual tests miss.
    """

    def test_full_pipeline(self, v_valley_path, expectations):
        """Run the complete pipeline on a fresh Grid and validate all outputs."""
        cross_slope = expectations["cross_slope"]
        channel_col = expectations["channel_col"]
        pixel_size = expectations["pixel_size"]
        expected_slope = expectations["slope"]

        # -- Load from raster (fresh Grid, no shared state) --
        grid = Grid.from_raster(v_valley_path, "dem")
        assert not grid._crs_is_geographic(), "Should detect UTM as non-geographic"

        # -- Flow routing --
        grid.flowdir("dem", out_name="fdir", dirmap=DIRMAP, routing="d8")
        grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

        # -- Channel delineation --
        acc_mask = grid.acc > NROWS
        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        # -- HAND/DTND/AZND --
        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
            routing="d8",
        )

        margin = 5
        edge = 5

        # HAND should be positive and bounded
        valid_hand = grid.hand[~np.isnan(grid.hand)]
        assert np.all(valid_hand >= 0), "HAND has negative values"
        max_expected_hand = cross_slope * (NCOLS / 2) * pixel_size
        assert np.max(valid_hand) < max_expected_hand * 1.5, (
            f"Max HAND={np.max(valid_hand):.1f}, expected < {max_expected_hand * 1.5:.1f}"
        )

        # DTND should be positive and bounded (not haversine garbage)
        valid_dtnd = grid.dtnd[~np.isnan(grid.dtnd)]
        assert np.max(valid_dtnd) < (NCOLS / 2) * pixel_size * 1.5, (
            f"Max DTND={np.max(valid_dtnd):.1f}m — likely haversine-on-UTM bug"
        )

        # HAND = cross_slope * DTND on a sample row
        sample_row = NROWS // 2
        w_hand = grid.hand[sample_row, edge : channel_col - margin]
        w_dtnd = grid.dtnd[sample_row, edge : channel_col - margin]
        tol = cross_slope * pixel_size
        np.testing.assert_allclose(
            w_hand,
            cross_slope * w_dtnd,
            atol=tol,
            err_msg="HAND != cross_slope * DTND on sample row",
        )

        # -- Slope/aspect --
        grid.slope_aspect("dem")

        west_slope = grid.slope[edge:-edge, edge : channel_col - 3]
        assert west_slope == pytest.approx(expected_slope, abs=0.001), (
            f"Slope mismatch: mean={np.mean(west_slope):.5f}, expected={expected_slope:.5f}"
        )

        west_aspect = grid.aspect[edge:-edge, edge : channel_col - 3]
        assert west_aspect == pytest.approx(
            expectations["aspect_west_side"], abs=0.5
        ), f"West aspect mismatch: mean={np.mean(west_aspect):.2f}"

        # -- Hillslope classification --
        grid.compute_hillslope("fdir", grid.channel_mask, grid.bank_mask, dirmap=DIRMAP)
        present = set(np.unique(grid.hillslope[grid.hillslope > 0]))
        assert present == {1, 2, 3, 4}, (
            f"Expected hillslope types {{1,2,3,4}}, got {present}"
        )


class TestGeographicRegression:
    """Verify that the UTM changes don't break geographic CRS behavior.

    These run the same tests as test_hillslope.py's core workflow on the
    existing geographic DEM to confirm the if/else branch doesn't alter
    existing behavior.
    """

    @pytest.fixture(scope="class")
    def geo_grid(self):
        """Load the existing geographic test DEM."""
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dem_path = os.path.join(current_dir, "..", "data", "dem.tif")
        grid = Grid.from_raster(dem_path, "dem")
        return grid

    @pytest.fixture(scope="class")
    def geo_processed(self, geo_grid):
        """Process the geographic DEM through full pipeline."""
        grid = geo_grid
        grid.resolve_flats("dem", out_name="inflated_dem")
        grid.flowdir("inflated_dem", out_name="fdir", dirmap=DIRMAP, routing="d8")
        grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")
        return grid

    def test_slope_aspect_runs(self, geo_grid):
        """slope_aspect still works on geographic CRS."""
        grid = geo_grid
        grid.slope_aspect("dem")
        assert grid.slope is not None
        assert grid.aspect is not None
        valid_slope = grid.slope[~np.isnan(grid.slope)]
        assert np.all(valid_slope >= 0)

    def test_compute_hand_runs(self, geo_processed):
        """compute_hand still works on geographic CRS."""
        grid = geo_processed
        acc_mask = grid.acc > 100
        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)
        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
            routing="d8",
        )
        assert grid.hand is not None
        valid_hand = grid.hand[~np.isnan(grid.hand)]
        assert np.all(valid_hand >= 0)
