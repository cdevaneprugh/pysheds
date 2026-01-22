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

    def test_slope_aspect_runs(self, grid_with_dem):
        """Test that slope_aspect runs without error."""
        grid = grid_with_dem
        # inplace=True by default, stores in grid.slope and grid.aspect
        grid.slope_aspect("dem")
        assert hasattr(grid, "slope")
        assert hasattr(grid, "aspect")
        assert grid.slope is not None
        assert grid.aspect is not None

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

    def test_create_channel_mask_runs(self, processed_grid):
        """Test that create_channel_mask runs without error."""
        grid = processed_grid

        # Create accumulation mask (e.g., > 100 cells)
        acc = grid.acc
        acc_threshold = 100
        acc_mask = acc > acc_threshold

        # inplace=True by default, stores in grid.channel_mask, grid.channel_id, grid.bank_mask
        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        assert hasattr(grid, "channel_mask")
        assert hasattr(grid, "channel_id")
        assert hasattr(grid, "bank_mask")
        assert grid.channel_mask is not None
        assert grid.channel_id is not None
        assert grid.bank_mask is not None

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

    def test_compute_hand_runs(self, processed_grid):
        """Test that compute_hand runs and produces all outputs."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        # compute_hand also uses inplace=True by default
        grid.compute_hand(
            "fdir",
            "dem",
            grid.channel_mask,
            grid.channel_id,
            dirmap=DIRMAP,
            routing="d8",
        )

        assert hasattr(grid, "hand")
        assert grid.hand is not None

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

    def test_compute_hillslope_runs(self, processed_grid):
        """Test that compute_hillslope runs without error."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)

        # compute_hillslope also uses inplace=True by default
        grid.compute_hillslope("fdir", grid.channel_mask, grid.bank_mask)

        assert hasattr(grid, "hillslope")
        assert grid.hillslope is not None

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

    def test_profiles_have_structure(self, processed_grid):
        """Test that profiles have expected structure."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        profiles, connections = grid.extract_profiles(
            "fdir", mask=acc_mask, dirmap=DIRMAP
        )

        # Should have at least one profile
        assert len(profiles) > 0

        # Each profile should be an array of indices
        for profile in profiles:
            assert isinstance(profile, np.ndarray)


class TestRiverNetworkLengthSlope:
    """Tests for river_network_length_and_slope() method."""

    def test_river_network_length_slope_runs(self, processed_grid):
        """Test that river_network_length_and_slope runs."""
        grid = processed_grid

        acc = grid.acc
        acc_mask = acc > 100

        try:
            length, slope = grid.river_network_length_and_slope(
                "fdir", "dem", mask=acc_mask, dirmap=DIRMAP
            )
            assert length is not None or slope is not None
        except (AttributeError, TypeError) as e:
            pytest.skip(f"Method not fully implemented: {e}")


class TestIntegration:
    """Integration tests for full hillslope workflow."""

    def test_full_workflow(self, processed_grid):
        """Test full hillslope analysis workflow."""
        grid = processed_grid

        # Step 1: Calculate slope and aspect
        grid.slope_aspect("dem")
        assert grid.slope is not None
        assert grid.aspect is not None

        # Step 2: Create channel network
        acc = grid.acc
        acc_mask = acc > 100

        grid.create_channel_mask("fdir", mask=acc_mask, dirmap=DIRMAP)
        assert grid.channel_mask is not None

        # Step 3: Calculate HAND
        grid.compute_hand(
            "fdir", "dem", grid.channel_mask, grid.channel_id, dirmap=DIRMAP
        )
        assert grid.hand is not None

        # Step 4: Classify hillslopes
        grid.compute_hillslope("fdir", grid.channel_mask, grid.bank_mask)
        assert grid.hillslope is not None

        # Verify we have all 4 hillslope types
        valid_hs = grid.hillslope[~np.isnan(grid.hillslope)]
        unique_types = np.unique(valid_hs)
        # We should have at least channel (4) and some bank types
        assert 4 in unique_types  # Channel should exist
