"""
Tests for hillslope-specific methods in pgrid.py

These methods are Swenson's additions for representative hillslope analysis:
- slope_aspect() - Calculate slope and aspect from DEM
- create_channel_mask() - Create channel mask, IDs, and bank classification
- compute_hand() - Extended HAND with DTND, AZND, drainage_id
- compute_hillslope() - Hillslope classification (L/R bank, headwater)
- extract_profiles() - River segment extraction
- river_network_length_and_slope() - Network metrics
"""

import pytest
import numpy as np
import os

from pysheds.pgrid import Grid


# Test data paths
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../data"))
DEM_PATH = os.path.join(DATA_DIR, "dem.tif")

# D8 direction map
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


@pytest.fixture
def grid_with_dem():
    """Create a Grid with DEM loaded."""
    grid = Grid.from_raster(DEM_PATH, "dem")
    return grid


@pytest.fixture
def processed_grid(grid_with_dem):
    """Create a Grid with processed flow direction and accumulation."""
    grid = grid_with_dem

    # Fill depressions and resolve flats (stores as grid.inflated_dem)
    grid.resolve_flats("dem", out_name="inflated_dem")

    # Calculate flow direction (D8) (stores as grid.fdir)
    grid.flowdir("inflated_dem", out_name="fdir", dirmap=DIRMAP, routing="d8")

    # Calculate accumulation (stores as grid.acc)
    grid.accumulation("fdir", out_name="acc", dirmap=DIRMAP, routing="d8")

    return grid


class TestSlopeAspect:
    """Tests for slope_aspect() method."""

    def test_slope_aspect_output_shape(self, grid_with_dem):
        """Test that slope_aspect outputs have correct shape."""
        grid = grid_with_dem
        grid.slope_aspect("dem")

        dem_shape = grid.dem.shape
        assert grid.slope.shape == dem_shape
        assert grid.aspect.shape == dem_shape

    def test_slope_non_negative(self, grid_with_dem):
        """Test that slope values are non-negative."""
        grid = grid_with_dem
        grid.slope_aspect("dem")

        # Slope should be non-negative (ignoring NaN)
        valid_slope = grid.slope[~np.isnan(grid.slope)]
        assert np.all(valid_slope >= 0)

    def test_aspect_range(self, grid_with_dem):
        """Test that aspect values are in valid range."""
        grid = grid_with_dem
        grid.slope_aspect("dem")

        # Aspect is converted to [0, 360] by the method
        valid_aspect = grid.aspect[~np.isnan(grid.aspect)]
        assert np.all(valid_aspect >= 0)
        assert np.all(valid_aspect <= 360)


class TestChannelMask:
    """Tests for create_channel_mask() method."""

    def test_channel_mask_is_binary(self, processed_grid):
        """Test that channel_mask is binary (0 or 1)."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        valid_mask = grid.channel_mask[~np.isnan(grid.channel_mask)]
        unique_values = np.unique(valid_mask)
        # Should only contain 0 and/or 1
        assert np.all(np.isin(unique_values, [0, 1]))

    def test_bank_mask_values(self, processed_grid):
        """Test that bank_mask has valid values (+1, 0, -1)."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        valid_bank = grid.bank_mask[~np.isnan(grid.bank_mask)]
        # Bank mask should be -1 (left), 0 (channel), or +1 (right)
        assert np.all(np.isin(valid_bank, [-1, 0, 1]))


class TestComputeHand:
    """Tests for compute_hand() method (extended version)."""

    def test_hand_non_negative(self, processed_grid):
        """Test that HAND values are non-negative."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
        )

        valid_hand = grid.hand[~np.isnan(grid.hand)]
        # HAND (Height Above Nearest Drainage) should be >= 0
        assert np.all(valid_hand >= 0)


class TestComputeHillslope:
    """Tests for compute_hillslope() method."""

    def test_hillslope_valid_values(self, processed_grid):
        """Test that hillslope classification has valid values."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        grid.compute_hillslope("fdir", grid.channel_mask, grid.bank_mask)

        valid_hs = grid.hillslope[~np.isnan(grid.hillslope)]
        # Hillslope values: 0=unclassified, 1=headwater, 2=right bank, 3=left bank, 4=channel
        assert np.all(np.isin(valid_hs, [0, 1, 2, 3, 4]))
        # Ensure we have at least some classified values (not all zeros)
        assert np.sum(valid_hs > 0) > 0


class TestExtractProfiles:
    """Tests for extract_profiles() method."""

    def test_extract_profiles_runs(self, processed_grid):
        """Test that extract_profiles runs without error."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        # extract_profiles returns (profiles_list, connections_dict)
        result = grid.extract_profiles("fdir", mask=acc_mask, dirmap=DIRMAP)

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

        profiles, connections = result
        assert isinstance(profiles, list)
        assert isinstance(connections, dict)

    def test_profiles_are_channel_pixels(self, processed_grid):
        """Test that profile indices correspond to channel pixels.

        Each profile should contain only flat indices that fall on pixels
        where accumulation exceeds the threshold (i.e., actual channel pixels).
        """
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        profiles, connections = grid.extract_profiles(
            "fdir", mask=acc_mask, dirmap=DIRMAP
        )

        assert len(profiles) > 0

        # Flatten the acc_mask to match flat index format
        acc_mask_flat = np.asarray(acc_mask).ravel()

        total_profile_pixels = 0
        on_channel_count = 0
        for profile in profiles:
            assert len(profile) >= 1, "Empty profile array"
            on_channel_count += np.sum(acc_mask_flat[profile])
            total_profile_pixels += len(profile)

        # Nearly all profile pixels should be on the channel network.
        # Endpoint/junction pixels may fall just below the threshold.
        on_channel_frac = on_channel_count / total_profile_pixels
        assert on_channel_frac > 0.9, (
            f"Only {on_channel_frac:.1%} of profile pixels are on the channel "
            f"network ({on_channel_count}/{total_profile_pixels})"
        )

        # Profiles should cover a meaningful fraction of channel pixels
        total_channel_pixels = np.sum(acc_mask_flat)
        coverage = total_profile_pixels / total_channel_pixels
        assert coverage > 0.5, (
            f"Profiles cover only {coverage:.1%} of channel pixels "
            f"({total_profile_pixels}/{total_channel_pixels})"
        )


class TestRiverNetworkLengthSlope:
    """Tests for river_network_length_and_slope() method."""

    def test_river_network_length_slope_runs(self, processed_grid):
        """Test that river_network_length_and_slope runs."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        # Method signature: river_network_length_and_slope(fdir, mask, dirmap=None, ...)
        # Returns a dictionary with keys: length, slope, mch_length, mch_slope, etc.
        result = grid.river_network_length_and_slope("fdir", acc_mask, dirmap=DIRMAP)

        assert isinstance(result, dict)
        assert "length" in result
        assert "slope" in result
        assert result["length"] is not None
        assert result["slope"] is not None


class TestIntegration:
    """Integration test: full pipeline chains without state-passing bugs."""

    def test_full_workflow(self, processed_grid):
        """Run complete pipeline and verify hillslope classification output.

        Individual methods are tested by their own classes. This test verifies
        that the full chain produces a valid hillslope classification with all
        expected types present — catching state-passing bugs between stages.
        """
        grid = processed_grid

        grid.slope_aspect("dem")

        acc_mask = grid.acc > 100
        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)
        grid.compute_hand(
            "fdir", "dem", grid.channel_mask, grid.channel_id, dirmap=DIRMAP
        )
        grid.compute_hillslope("fdir", grid.channel_mask, grid.bank_mask)

        valid_hs = grid.hillslope[~np.isnan(grid.hillslope)]
        unique_types = set(np.unique(valid_hs))
        # Should have channel (4) and at least one bank/headwater type
        assert 4 in unique_types, f"Channel type (4) missing from {unique_types}"
        assert len(unique_types) >= 3, (
            f"Expected at least 3 hillslope types, got {unique_types}"
        )
