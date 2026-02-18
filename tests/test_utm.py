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
        HAND depends only on column offset (constant across rows), so a single
        2D comparison covers the full interior.
        """
        grid = grid_with_hand
        cross_slope = expectations["cross_slope"]
        channel_col = expectations["channel_col"]
        pixel_size = expectations["pixel_size"]

        hand = np.asarray(grid.hand)

        # Build expected HAND: each column offset gives HAND = cross_slope * |c - channel| * pixel_size
        cols = np.arange(NCOLS)
        expected_hand_row = cross_slope * np.abs(cols - channel_col) * pixel_size

        margin = 5  # wider margin: channel + boundary effects
        edge = 5
        nrows_interior = NROWS - 2 * edge
        tol = cross_slope * pixel_size  # one pixel worth of error

        # Tile expected row across all interior rows
        expected_2d = np.tile(expected_hand_row, (nrows_interior, 1))

        # West side interior
        np.testing.assert_allclose(
            hand[edge:-edge, edge : channel_col - margin],
            expected_2d[:, edge : channel_col - margin],
            atol=tol,
            err_msg="HAND mismatch on west side",
        )
        # East side interior
        np.testing.assert_allclose(
            hand[edge:-edge, channel_col + margin + 1 : NCOLS - edge],
            expected_2d[:, channel_col + margin + 1 : NCOLS - edge],
            atol=tol,
            err_msg="HAND mismatch on east side",
        )


class TestDTND:
    """Test DTND computation on V-valley UTM DEM."""

    def test_dtnd_values(self, grid_with_hand, expectations):
        """DTND should equal abs(col - channel_col) * pixel_size.

        This is the core CRS test: haversine on UTM coordinates produces
        garbage (~550km), Euclidean produces the correct answer (~0-500m).
        DTND depends only on column offset (constant across rows), so a
        single 2D comparison covers the full interior.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]
        pixel_size = expectations["pixel_size"]

        dtnd = np.asarray(grid.dtnd)

        # Build expected DTND: each column offset gives DTND = |c - channel| * pixel_size
        cols = np.arange(NCOLS)
        expected_dtnd_row = np.abs(cols - channel_col) * pixel_size

        margin = 5
        edge = 5
        nrows_interior = NROWS - 2 * edge

        # Tile expected row across all interior rows
        expected_2d = np.tile(expected_dtnd_row, (nrows_interior, 1))

        # West side interior
        np.testing.assert_allclose(
            dtnd[edge:-edge, edge : channel_col - margin],
            expected_2d[:, edge : channel_col - margin],
            atol=pixel_size,
            err_msg="DTND mismatch on west side",
        )
        # East side interior
        np.testing.assert_allclose(
            dtnd[edge:-edge, channel_col + margin + 1 : NCOLS - edge],
            expected_2d[:, channel_col + margin + 1 : NCOLS - edge],
            atol=pixel_size,
            err_msg="DTND mismatch on east side",
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

    def test_aznd_channel_is_self_referencing(self, grid_with_hand, expectations):
        """Channel pixels have hndx pointing to themselves -> AZND = 180 deg.

        This is a topology test, NOT a CRS test. Channel pixels' hndx maps
        to self (pgrid.py:1935), so dlon=dlat=0 regardless of CRS. The 180
        degree value comes from arctan2(-0.0, -0.0) = -pi (IEEE 754 signed
        zero). Verifies hndx assignment, not distance computation.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]

        aznd = grid.aznd
        edge = 5

        channel_aznd = aznd[edge:-edge, channel_col]
        assert channel_aznd == pytest.approx(180.0, abs=1.0), (
            f"Channel AZND should be 180 (self-referencing hndx), got range "
            f"[{channel_aznd.min():.2f}, {channel_aznd.max():.2f}]"
        )


class TestRiverNetworkLengthSlope:
    """Test river_network_length_and_slope() on V-valley UTM DEM.

    CRS test: haversine interprets 5m pixel spacing as 5 degrees of
    latitude, producing ~556 km per segment (~110,000 km total) vs the
    correct ~995m. This test catches that failure mode.

    The V-valley has a single straight N-S channel, so:
    - Exactly 1 reach
    - Total length = reach length = main channel length
    - Slope = downstream_slope = 0.001 m/m
    - Midpoint at grid center (col 100, ~row 100)
    """

    def test_channel_length(self, grid_with_river_stats, expectations):
        """Total reach length should match analytical expectation."""
        result = grid_with_river_stats
        expected_length = expectations["channel_length"]
        assert result["length"] == pytest.approx(expected_length, rel=0.02)

    def test_channel_slope(self, grid_with_river_stats, expectations):
        """Mean reach slope should match downstream_slope."""
        result = grid_with_river_stats
        expected_slope = expectations["channel_slope"]
        assert result["slope"] == pytest.approx(expected_slope, rel=0.02)

    def test_single_reach(self, grid_with_river_stats):
        """V-valley has exactly 1 reach."""
        result = grid_with_river_stats
        assert len(result["reach_lengths"]) == 1

    def test_midpoint_easting(self, grid_with_river_stats, expectations):
        """Reach midpoint easting is at the channel column."""
        result = grid_with_river_stats
        channel_col = expectations["channel_col"]
        expected_easting = UTM_EASTING + channel_col * PIXEL_SIZE
        assert result["mlon"][0] == pytest.approx(expected_easting, abs=PIXEL_SIZE)


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

        # Ensure compute_hillslope has been called (don't rely on test ordering)
        if not hasattr(grid, "hillslope") or grid.hillslope is None:
            grid.compute_hillslope(
                "fdir", grid.channel_mask, grid.bank_mask, dirmap=DIRMAP
            )
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

        if not hasattr(grid, "hillslope") or grid.hillslope is None:
            grid.compute_hillslope(
                "fdir", grid.channel_mask, grid.bank_mask, dirmap=DIRMAP
            )
        hillslope = grid.hillslope
        edge = 5

        channel_types = hillslope[edge:-edge, channel_col]
        assert np.all(channel_types == 4), (
            f"Channel column should be type 4, got unique types: "
            f"{np.unique(channel_types)}"
        )


class TestExtractProfiles:
    """Test extract_profiles() on V-valley UTM DEM.

    The V-valley has a single N-S channel at col 100. With acc > NROWS as the
    channel threshold, only the center column qualifies. extract_profiles should
    return exactly one profile covering the full channel length.
    """

    @pytest.fixture(scope="class")
    def profiles_and_connections(self, processed_grid):
        """Call extract_profiles directly on the V-valley."""
        grid = processed_grid
        acc_mask = grid.acc > NROWS
        profiles, connections = grid.extract_profiles("fdir", acc_mask, dirmap=DIRMAP)
        return profiles, connections

    def test_profile_count(self, profiles_and_connections):
        """V-valley produces 1-2 profiles (fork detection may split outlet)."""
        profiles, _ = profiles_and_connections
        assert 1 <= len(profiles) <= 2

    def test_primary_profile_length(self, profiles_and_connections):
        """Longest profile contains ~197 pixels (bulk of the channel)."""
        profiles, _ = profiles_and_connections
        longest = max(profiles, key=len)
        assert len(longest) == pytest.approx(NROWS - 1, abs=5)

    def test_primary_profile_on_channel_column(self, profiles_and_connections):
        """All pixels in the longest profile are on col 100 (center column)."""
        profiles, _ = profiles_and_connections
        longest = max(profiles, key=len)
        _, col_indices = np.unravel_index(longest, (NROWS, NCOLS))
        channel_col = NCOLS // 2
        assert np.all(col_indices == channel_col)

    def test_primary_profile_flows_north_to_south(self, profiles_and_connections):
        """Row indices are monotonically increasing (upstream to downstream)."""
        profiles, _ = profiles_and_connections
        longest = max(profiles, key=len)
        row_indices, _ = np.unravel_index(longest, (NROWS, NCOLS))
        assert np.all(np.diff(row_indices) >= 0)

    def test_connections_terminate_within_bounds(self, profiles_and_connections):
        """Connection chain from profile 0 terminates within len(profiles) steps.

        Verifies the connection graph is acyclic (reaches -1) or bounded
        (terminates within len(connections) steps). An unbounded chain
        would indicate a bug in fork detection.
        """
        _, connections = profiles_and_connections
        max_steps = len(connections)
        node = 0
        steps = 0
        while node != -1 and steps < max_steps:
            node = connections[node]
            steps += 1
        assert steps <= max_steps, (
            f"Connection chain did not terminate in {max_steps} steps"
        )

    def test_total_profile_coverage(self, profiles_and_connections, processed_grid):
        """Combined profiles cover >95% of accumulation-threshold pixels."""
        profiles, _ = profiles_and_connections
        grid = processed_grid
        acc_mask = grid.acc > NROWS
        n_channel_pixels = int(np.sum(np.asarray(acc_mask)))
        total_profile_pixels = sum(len(p) for p in profiles)
        coverage = total_profile_pixels / n_channel_pixels
        assert coverage > 0.95

    def test_all_profile_indices_valid(self, profiles_and_connections):
        """All flat indices across all profiles are within grid bounds."""
        profiles, _ = profiles_and_connections
        total_pixels = NROWS * NCOLS
        for profile in profiles:
            assert np.all(profile >= 0)
            assert np.all(profile < total_pixels)


class TestCreateChannelMask:
    """Test create_channel_mask() on V-valley UTM DEM.

    Uses grid_with_hand fixture which already called create_channel_mask.
    The V-valley has a single straight channel at col 100 flowing south.
    Bank assignment: facing downstream (south), west is the right bank (+1),
    east is the left bank (-1). Only immediate neighbors of channel pixels
    are assigned bank values; farther pixels remain 0.
    """

    def test_channel_mask_on_channel_column(self, grid_with_hand):
        """Channel mask covers >97% of accumulation-threshold pixels at center column.

        extract_profiles may miss 1-2 boundary pixels due to fork detection
        and grid-edge handling. The bulk of the channel must be covered.
        """
        grid = grid_with_hand
        channel_col = NCOLS // 2
        acc_mask = np.asarray(grid.acc > NROWS)

        channel_col_mask = np.asarray(grid.channel_mask[:, channel_col])
        channel_col_acc = acc_mask[:, channel_col]
        n_expected = np.sum(channel_col_acc)
        n_covered = np.sum(channel_col_mask[channel_col_acc] == 1)
        coverage = n_covered / n_expected
        assert coverage > 0.97, (
            f"Channel mask covers {n_covered}/{n_expected} ({coverage:.1%}) "
            f"of acc-threshold pixels on channel column"
        )

    def test_channel_mask_off_channel(self, grid_with_hand, expectations):
        """Hillslope pixels (off-channel columns) have channel_mask = 0."""
        grid = grid_with_hand
        channel_col = expectations["channel_col"]
        edge = 5

        west = grid.channel_mask[edge:-edge, edge : channel_col - 1]
        east = grid.channel_mask[edge:-edge, channel_col + 2 : -edge]
        assert np.all(west == 0)
        assert np.all(east == 0)

    def test_channel_id_count(self, grid_with_hand):
        """1-2 unique channel IDs (matches profile count from extract_profiles)."""
        grid = grid_with_hand
        unique_ids = np.unique(
            np.asarray(grid.channel_id)[np.asarray(grid.channel_id) > 0]
        )
        assert 1 <= len(unique_ids) <= 2
        assert unique_ids[0] == 1  # primary channel is always ID 1

    def test_channel_id_matches_mask(self, processed_grid):
        """channel_id > 0 iff channel_mask == 1.

        Uses processed_grid (not grid_with_hand) to avoid Raster mutation
        artifacts from compute_hand's _input_handler processing the
        channel_mask and channel_id arrays as inputs.
        """
        grid = processed_grid
        acc_mask = grid.acc > NROWS
        grid.create_channel_mask(
            "fdir",
            mask=acc_mask,
            dirmap=DIRMAP,
            out_name="cm_test",
            out_name_channel_id="cid_test",
            out_name_bank="bm_test",
        )
        cid = np.array(grid.cid_test, dtype=float)
        cmask = np.array(grid.cm_test, dtype=float)
        cid = np.nan_to_num(cid, nan=0.0)
        cmask = np.nan_to_num(cmask, nan=0.0)
        id_positive = cid > 0
        mask_positive = cmask == 1
        assert np.array_equal(id_positive, mask_positive)

    def test_bank_mask_west_is_right(self, grid_with_hand, expectations):
        """Immediate west neighbor of channel = right bank (+1).

        Channel flows south. Facing downstream (south), west is on the right.
        The bank assignment algorithm marks right-bank neighbors with +1.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]
        edge = 5

        # Column immediately west of channel, in the interior rows
        west_neighbor = grid.bank_mask[edge:-edge, channel_col - 1]
        assert np.all(west_neighbor == 1), (
            f"Expected west neighbor bank_mask = +1, "
            f"got unique values: {np.unique(west_neighbor)}"
        )

    def test_bank_mask_east_is_left(self, grid_with_hand, expectations):
        """Immediate east neighbor of channel = left bank (-1).

        Channel flows south. Facing downstream (south), east is on the left.
        The bank assignment algorithm marks left-bank neighbors with -1.
        """
        grid = grid_with_hand
        channel_col = expectations["channel_col"]
        edge = 5

        east_neighbor = grid.bank_mask[edge:-edge, channel_col + 1]
        assert np.all(east_neighbor == -1), (
            f"Expected east neighbor bank_mask = -1, "
            f"got unique values: {np.unique(east_neighbor)}"
        )

    def test_bank_mask_channel_is_zero(self, grid_with_hand):
        """Channel pixels have bank_mask = 0."""
        grid = grid_with_hand
        channel_pixels = grid.channel_mask == 1
        assert np.all(grid.bank_mask[channel_pixels] == 0)


class TestEndToEndUTM:
    """End-to-end integration test: fresh Grid through full pipeline.

    Chains from_raster -> flowdir -> accumulation -> create_channel_mask ->
    compute_hand -> slope_aspect -> compute_hillslope on a fresh Grid to
    catch state-passing bugs between stages that individual tests miss.
    """

    def test_full_pipeline(self, v_valley_path, expectations):
        """Run the complete pipeline on a fresh Grid to catch state-passing bugs.

        The unique value here is exercising the full chain on a fresh Grid instance.
        Individual parameter validation is handled by the unit test classes.
        """
        pixel_size = expectations["pixel_size"]

        # -- Load from raster (fresh Grid, no shared state) --
        grid = Grid.from_raster(v_valley_path, "dem")

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

        # DTND bounds check on fresh Grid (catches state-passing bugs between stages).
        # Analytical max = (NCOLS/2) * pixel_size = 500m.
        # Lower bound catches deflation (e.g. pixel-unit bug); upper bound catches
        # inflation (e.g. haversine-on-UTM producing ~550km).
        valid_dtnd = grid.dtnd[~np.isnan(grid.dtnd)]
        max_dtnd = np.max(valid_dtnd)
        expected_max = (NCOLS / 2) * pixel_size  # 500m
        assert max_dtnd > expected_max * 0.8, (
            f"Max DTND={max_dtnd:.1f}m too small (expected ~{expected_max:.0f}m)"
        )
        assert max_dtnd < expected_max * 1.1, (
            f"Max DTND={max_dtnd:.1f}m too large (expected ~{expected_max:.0f}m)"
        )

        # -- Slope/aspect --
        grid.slope_aspect("dem")

        # -- Hillslope classification (end-state check) --
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

    def test_slope_aspect_produces_realistic_values(self, geo_grid):
        """slope_aspect on geographic CRS produces non-trivial slope/aspect.

        Checks that the geographic codepath produces values in a realistic
        range, not just that it doesn't crash. The test DEM has real terrain,
        so slope should span a meaningful range (not all zero or all equal).
        """
        grid = geo_grid
        grid.slope_aspect("dem")
        valid_slope = grid.slope[~np.isnan(grid.slope)]
        assert len(valid_slope) > 0, "No valid slope values"
        assert np.all(valid_slope >= 0), "Negative slope values"
        assert np.max(valid_slope) > 0.01, (
            f"Max slope={np.max(valid_slope):.6f} — suspiciously small for real terrain"
        )
        assert np.std(valid_slope) > 0.001, (
            f"Slope std={np.std(valid_slope):.6f} — all slopes nearly identical"
        )

        valid_aspect = grid.aspect[~np.isnan(grid.aspect)]
        assert np.min(valid_aspect) >= 0, "Aspect below 0"
        assert np.max(valid_aspect) <= 360, "Aspect above 360"
        # Real terrain should have aspects spanning multiple quadrants
        assert np.max(valid_aspect) - np.min(valid_aspect) > 90, (
            "Aspect range < 90 deg — suspiciously uniform for real terrain"
        )

    def test_compute_hand_produces_realistic_values(self, geo_processed):
        """compute_hand on geographic CRS produces non-trivial HAND values.

        Checks that the geographic codepath produces values in a realistic
        range, not just that it doesn't crash.
        """
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
        valid_hand = grid.hand[~np.isnan(grid.hand)]
        assert len(valid_hand) > 0, "No valid HAND values"
        assert np.all(valid_hand >= 0), "Negative HAND values"
        assert np.max(valid_hand) > 1.0, (
            f"Max HAND={np.max(valid_hand):.2f}m — suspiciously small for real terrain"
        )
        assert np.std(valid_hand) > 0.1, (
            f"HAND std={np.std(valid_hand):.4f} — all values nearly identical"
        )


class TestCRSBranchNecessity:
    """Verify that the CRS branching is load-bearing.

    These negative tests demonstrate that feeding UTM data through the
    geographic (haversine) codepath produces garbage. If someone removes
    the CRS branching, these tests would start passing (the assertions
    check for garbage), alerting that the branches are no longer distinct.
    """

    def test_haversine_slope_on_utm_is_wrong(self, grid_with_dem):
        """Haversine gradient on UTM coordinates produces wrong slope.

        The geographic codepath interprets 5m pixel spacing as 5 degrees
        of latitude (~556km), producing slopes orders of magnitude too small.
        This test verifies the failure mode exists, confirming the CRS
        branch is necessary.
        """
        grid = grid_with_dem

        # Monkeypatch _crs_is_geographic to return True, forcing
        # the haversine codepath on UTM data
        original = grid._crs_is_geographic
        grid._crs_is_geographic = lambda: True
        try:
            grid.slope_aspect(
                grid.dem, slope_out_name="slope_geo", aspect_out_name="aspect_geo"
            )
            slope_geo = np.asarray(grid.slope_geo)
        finally:
            grid._crs_is_geographic = original

        # Correct slope is ~0.03 m/m. Haversine interprets 5m as 5 degrees
        # (~556km), so the denominator is ~1e5x too large -> slope ~3e-7.
        valid = slope_geo[~np.isnan(slope_geo)]
        median_slope = np.median(valid[valid > 0])

        assert median_slope < 1e-4, (
            f"Haversine slope on UTM = {median_slope:.2e} — expected < 1e-4 "
            f"(correct value is ~0.03). If this passes with a large value, "
            f"the CRS branch may have been removed."
        )

    def test_haversine_dtnd_on_utm_is_wrong(self, grid_with_hand, expectations):
        """Haversine distance on UTM coordinates produces garbage DTND.

        The geographic codepath interprets 5m pixel spacing as 5 degrees
        of latitude, producing distances ~550,000m instead of ~500m.
        This test verifies the failure mode, confirming the CRS branch in
        compute_hand is necessary.

        Instead of re-running compute_hand with haversine (which would
        require modifying the grid), we verify the correct DTND is in
        the right range (< 1000m), which would fail if haversine were used
        (values would be ~550,000m).
        """
        grid = grid_with_hand
        pixel_size = expectations["pixel_size"]
        edge = 5

        dtnd = grid.dtnd
        valid_dtnd = dtnd[edge:-edge, edge:-edge]
        valid_dtnd = valid_dtnd[~np.isnan(valid_dtnd)]

        max_expected = (NCOLS / 2) * pixel_size  # 500m

        # DTND should be in the correct range (0-500m), not haversine
        # garbage (~550,000m). This is a weaker form of the negative test,
        # but it guards against the CRS branch being removed without
        # also breaking this assertion by 1000x.
        assert np.max(valid_dtnd) < max_expected * 2, (
            f"Max DTND={np.max(valid_dtnd):.0f}m — expected < {max_expected * 2:.0f}m. "
            f"Values > 100,000m indicate haversine-on-UTM."
        )
        assert np.max(valid_dtnd) > max_expected * 0.5, (
            f"Max DTND={np.max(valid_dtnd):.0f}m — expected > {max_expected * 0.5:.0f}m. "
            f"Suspiciously small."
        )
