# Synthetic V-Valley DEM for Phase A Testing

## Purpose

The synthetic V-valley DEM validates pysheds' handling of UTM CRS data. The existing test DEM (`data/dem.tif`) is geographic CRS (EPSG:4326, ~93m resolution) and exercises the haversine code path. This DEM is UTM (EPSG:32617, 1m resolution) -- the CRS that exposed bugs in `compute_hand()` and `_gradient_horn_1981()`.

The idea is simple: we need a DEM where we know the exact right answer for every derived parameter (slope, aspect, HAND, DTND), so we can tell if pysheds gives wrong results when processing UTM data. A V-shaped valley is the simplest geometry where all of these have closed-form solutions.

## Geometry

Two planar hillslopes meeting at a central north-south channel, with a slight downstream (north-to-south) gradient.

### Cross-section (west to east)

```
     \                    /
      \    cross_slope   /
       \                /
        \              /
         \            /
          \__________/   <-- channel (center column)
```

### Profile (north to south, along channel)

```
  North (row 0) ___
                    \___
                         \___
                              \___  South (row 999)
                                    <-- downstream_slope
```

## Elevation Formula

```
elev[r, c] = base_elevation
            + downstream_slope * (nrows - 1 - r) * pixel_size
            + cross_slope * abs(c - channel_col) * pixel_size
```

Where `channel_col = ncols // 2 = 500`.

Three additive terms, each with a distinct role:

### Term 1: base_elevation (default 50.0m)

The floor. The lowest point in the DEM is the south end of the channel at `[999, 500]`, which equals exactly `base_elevation`. Shifting this moves the whole surface up or down. It doesn't affect any derived quantity -- slope, aspect, HAND, and DTND are all relative.

### Term 2: downstream_slope * (nrows - 1 - r) * pixel_size

The N-to-S tilt along the channel. `(nrows - 1 - r)` maps row indices to "distance from south end":

- Row 0 (north): `1000 - 1 - 0 = 999` -> 999m from south -> max downstream elevation
- Row 999 (south): `1000 - 1 - 999 = 0` -> 0m from south -> base elevation

At default 0.001 m/m, the channel gains 0.999m over its 999m length -- a very gentle southward slope.

### Term 3: cross_slope * abs(c - channel_col) * pixel_size

The V-shape. `abs(c - 500)` is zero at the channel and increases linearly toward both edges:

- Col 500 (channel): `abs(500 - 500) = 0` -> no cross-valley contribution
- Col 0 (west ridge): `abs(0 - 500) = 500` -> 500m from channel
- Col 999 (east ridge): `abs(999 - 500) = 499` -> 499m from channel

At default 0.03 m/m, the west ridge stands 15.0m above the channel and the east ridge 14.97m.

### Slopes are the free parameters; ridge height is emergent

We define slopes and grid dimensions. Everything else falls out:

```
ridge_above_channel = cross_slope * (ncols // 2) * pixel_size
                    = 0.03 * 500 * 1.0 = 15.0m
```

The slopes analytically constrain every derived quantity:
- HAND = `cross_slope * distance_from_channel`
- DTND = `distance_from_channel`
- Slope = `sqrt(cross_slope^2 + downstream_slope^2)` (uniform everywhere)
- Aspect = `arctan2` of the two slope components (uniform on each side)
- D8 routing direction = always cardinal (guaranteed by `cross_slope >> downstream_slope`)

### Worked example: NE corner [0, 999]

```
elev[0, 999] = 50.0
             + 0.001 * (1000 - 1 - 0) * 1.0       = 0.999   (downstream term)
             + 0.03  * abs(999 - 500) * 1.0         = 14.97   (cross-valley term)
             = 65.969m
```

Compare to the NW corner [0, 0]: `50.0 + 0.999 + 0.03 * 500 = 65.999m`. The NW corner is 0.03m higher because the west ridge is 1 pixel farther from the channel (see Known Limitations).

## How Parameters Affect the Surface

| Parameter | Effect of increasing | Effect of decreasing |
|-----------|---------------------|---------------------|
| base_elevation | Shifts entire surface up. No effect on derived quantities. | Shifts down. |
| downstream_slope | Steeper N-to-S channel tilt. | Flatter channel. Zero creates flat cells for `resolve_flats`. |
| cross_slope | Steeper valley walls, higher ridges. | Flatter valley (more OSBS-like). |
| nrows/ncols | Larger domain. | Smaller domain. |
| pixel_size | Larger physical extent for same grid. | Finer sampling. |

The key design constraint: **cross_slope >> downstream_slope**. This ensures D8 routes straight east/west to the channel (never diagonally), giving clean flow paths with analytically predictable HAND and DTND.

## Cross Slope vs Downstream Slope

The naming follows the valley geometry:

- **Cross slope** (0.03 m/m): perpendicular to the channel. The channel runs N-S (along rows), so cross slope is the E-W gradient -- the valley walls rising away from the channel on both sides.
- **Downstream slope** (0.001 m/m): parallel to the channel. The N-S gradient along the channel itself.

## Default Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| nrows | 1000 | Matches NEON tile dimensions (1km at 1m) |
| ncols | 1000 | Same |
| pixel_size | 1.0 m | Matches NEON LIDAR resolution |
| cross_slope | 0.03 m/m | Middle of OSBS range (0.01-0.06) |
| downstream_slope | 0.001 m/m | Small enough for clean D8 routing to channel |
| base_elevation | 50.0 m | Within OSBS range (23-55m) |
| utm_easting | 400000 | Within OSBS tile grid |
| utm_northing | 3286000 | Within OSBS tile grid |
| EPSG | 32617 | UTM Zone 17N (NEON LIDAR CRS) |

### Why these slopes

- **cross_slope = 0.03**: Cardinal direction gradient (0.03) exceeds diagonal gradient (sqrt(0.03^2 + 0.001^2)/sqrt(2) = 0.0212), so D8 always routes straight across to the channel -- never diagonally. This gives clean, unambiguous flow paths.
- **downstream_slope = 0.001**: 3.3% of cross_slope. Large enough that `resolve_flats()` encounters no flat cells (every channel pixel has a unique downstream neighbor). Small enough to not dominate the aspect calculation (~1.9 deg offset from pure east/west).

## D8 Routing Behavior

The `cross_slope >> downstream_slope` constraint guarantees every non-channel pixel routes east or west to the channel, never north, south, or diagonally. Here's why.

D8 picks the neighbor with the steepest descent. For a pixel on the west side (c < 500), the candidates are:

| Neighbor | Drop | Distance | Gradient |
|----------|------|----------|----------|
| East (toward channel) | 0.030m | 1.0m | **0.030** |
| SE (diagonal) | ~0.030m | 1.414m | 0.0212 |
| South (downstream) | 0.001m | 1.0m | 0.001 |

East wins (0.030 > 0.0212 > 0.001). The diagonal is the only real competitor, and it loses because the sqrt(2) distance penalty dilutes the gradient. The condition is: `cross_slope > sqrt(cross_slope^2 + downstream_slope^2) / sqrt(2)`, which is easily satisfied when cross_slope >> downstream_slope (by 30x at the defaults).

Only the channel pixels (col 500) have no east-west gradient -- their sole descent is southward at 0.001 m/m.

Result: ridge-to-channel flow is purely E/W, channel flow is purely N/S.

## Analytical Expectations

### Slope (gradient magnitude)

```
slope = sqrt(cross_slope^2 + downstream_slope^2)
      = sqrt(0.03^2 + 0.001^2) = 0.030017 m/m  (~1.72 deg)
```

Uniform everywhere except within ~3 pixels of the channel (Horn stencil averages across the gradient discontinuity).

### Aspect (direction of steepest descent)

Computed via `arctan2(east_downhill, north_downhill)`:

- **West side** (c < channel_col): drains east and slightly south.
  `aspect = arctan2(+0.03, -0.001) = 91.91 deg` (slightly south of due east)
- **East side** (c > channel_col): drains west and slightly south.
  `aspect = arctan2(-0.03, -0.001) = 268.09 deg` (slightly south of due west)
- **Channel** (c = channel_col): only downstream component.
  `aspect = 180 deg` (due south)

### HAND (Height Above Nearest Drainage)

HAND is defined as: the elevation difference between a pixel and the stream pixel it drains to.

In the V-valley, every pixel drains straight east or west to the channel at the same row (guaranteed by D8 routing behavior above). So the pixel at `[r, c]` drains to `[r, channel_col]`:

```
HAND[r, c] = elev[r, c] - elev[r, channel_col]
           = cross_slope * abs(c - channel_col) * pixel_size
```

The downstream_slope terms cancel because both the pixel and its drainage point are in the same row. That's why HAND is **row-independent** -- it's purely a function of cross-valley distance.

Max at ridges (d = 500m): HAND = 15.0 m.

### DTND (Distance To Nearest Drainage)

DTND is defined as: the distance along the flow path from a pixel to the stream pixel it drains to.

Since D8 routes straight across (due to `cross_slope >> downstream_slope`), the flow path is a straight horizontal line:

```
DTND[r, c] = abs(c - channel_col) * pixel_size
```

### The HAND-DTND relationship

Combining the two: `HAND = cross_slope * DTND`. This is the hillslope profile -- a straight line with slope equal to `cross_slope`. On real terrain, this relationship is nonlinear and noisy. Here, it's a direct analytical consequence of the V-valley geometry.

## The GeoTIFF Row-vs-Northing Sign Issue

This section documents a confirmed bug in the OSBS pipeline's aspect calculation that the synthetic DEM is designed to catch.

### Convention

Standard GeoTIFFs store data top-to-bottom: row 0 is the northern edge, row index increases southward. The affine transform has negative y-resolution:

```
Affine(pixel_size, 0, utm_easting, 0, -pixel_size, utm_northing)
```

### Impact on gradient computation

`np.gradient(elev, pixel_size)` returns `dzdy, dzdx` where:
- `dzdx = d(elev)/d(col) = d(elev)/d(east)` -- correct as-is
- `dzdy = d(elev)/d(row) = d(elev)/d(south) = -d(elev)/d(north)` -- **sign is flipped**

### The pipeline bug (run_pipeline.py:1527)

```python
aspect = np.degrees(np.arctan2(-dzdx, -dzdy))
```

Standard aspect: `arctan2(east_downhill, north_downhill)` where:
- `east_downhill = -dzdx` -- correct
- `north_downhill = -d(elev)/d(north) = dzdy` -- pipeline uses `-dzdy` instead

The `-dzdy` double-negates: `-(-d(elev)/d(north)) = +d(elev)/d(north)` = uphill direction. This swaps North and South aspects. Correct expression: `arctan2(-dzdx, dzdy)`.

### What the V-valley catches

On the synthetic DEM with defaults:
- **West side:** correct = 91.91 deg, buggy = 88.09 deg (3.82 deg error)
- **East side:** correct = 268.09 deg, buggy = 271.91 deg (3.82 deg error)

The error is small because the cross-slope (0.03) dominates the downstream slope (0.001) -- inverting the tiny north component barely changes the angle. For terrain with comparable N-S and E-W gradients, the error would be much larger (up to 180 deg for a pure N-S slope). The V-valley primarily tests CRS math (haversine vs Euclidean); the aspect sign bug is confirmed by the affine analysis and detectable here as a systematic 3.82 deg offset.

### What is NOT affected

- **Slope magnitude**: `sqrt(dzdx^2 + dzdy^2)` is sign-independent
- **HAND**: elevation differences, not coordinates
- **DTND**: distance, always positive
- **Flow routing**: D8 compares neighbor elevations directly, not gradients

## Aspect Relevance for 1-Hillslope Configurations

For a 1-hillslope configuration with N elevation bins (no aspect binning -- our planned OSBS setup), the aspect sign bug doesn't affect output at all. Without aspect binning, no pixels get assigned to wrong hillslopes. You'd still want the correct aspect value for the single hillslope's circular mean (CTSM uses it for aspect-dependent insolation), but at OSBS slopes of 0.01-0.06 m/m the insolation correction is negligible regardless.

The detailed aspect analysis in this README is here for completeness and for anyone using 4-aspect configurations, where the N/S swap corrupts aspect-based binning and area fractions for all 16 hillslope columns. See `STATUS.md` problem #4 for the full corruption chain.

## Bug Detection Summary

| Bug | Detected by V-valley? | How |
|-----|----------------------|-----|
| Haversine DTND on UTM (pysheds `compute_hand`) | **Yes** | DTND values are garbage (haversine on meter-valued coordinates) |
| Haversine gradient spacing on UTM (pysheds `_gradient_horn_1981`) | **Yes** | Slope and aspect are wrong (spacing computed via haversine on meters) |
| N/S aspect swap (pipeline `arctan2(-dzdx, -dzdy)`) | Yes | Systematic 3.82 deg error (91.91 becomes 88.09, 268.09 becomes 271.91) |
| EDT vs flow-path DTND (conceptual) | **No** | V-valley gives the same answer for both (see Known Limitations) |
| Resolution effects (1m vs 4m subsampling) | Testable | Run at different `--pixel-size` values |

## Known Limitations

### 1-pixel asymmetry with even ncols

With `ncols = 1000` (indices 0-999), the true center is at 499.5 -- between columns 499 and 500. `ncols // 2 = 500` is half a pixel east of center:

- West side: columns 0-499 -> 500 pixels -> max distance 500m
- East side: columns 501-999 -> 499 pixels -> max distance 499m

This is internally consistent -- the elevation formula predicts the asymmetry, and pysheds should reproduce it exactly. The bugs we're testing for produce errors of a completely different character: haversine on UTM coordinates doesn't give you 501m instead of 500m, it gives you something astronomically wrong because it's interpreting meters as degrees.

For perfect symmetry, use odd ncols (e.g., `--ncols 999` or `--ncols 1001`).

### Cannot distinguish EDT-based vs flow-path-based DTND

In the V-valley, the hydrologically nearest stream pixel (the one each pixel drains to via D8) IS the geographically nearest stream pixel (the one closest in Euclidean distance). Both `scipy.ndimage.distance_transform_edt` and pysheds' flow-path-based DTND give the same answer.

On real terrain, these differ whenever a pixel is geographically close to a stream on the opposite side of a divide. Testing that distinction would require a more complex DEM with multiple drainage basins.

### Small aspect error from the sign bug

The aspect error on this DEM is only 3.82 degrees because `cross_slope` dominates `downstream_slope` by 30:1. For terrain with comparable N-S and E-W gradients, the error would be up to 180 degrees. The V-valley's primary purpose is catching the haversine-on-UTM bug (which produces garbage, not a subtle offset). The aspect sign bug is documented and detectable here, but not dramatically visible.

## Usage

```bash
# Generate with defaults
python data/synthetic_valley/generate_synthetic_dem.py

# Custom parameters
python data/synthetic_valley/generate_synthetic_dem.py --nrows 500 --ncols 500 --cross-slope 0.05

# Output goes to data/synthetic_valley/synthetic_valley_utm.tif by default
```

The `generate_v_valley()` function can also be called directly from pytest without file I/O:

```python
from data.synthetic_valley.generate_synthetic_dem import generate_v_valley

elev, expectations = generate_v_valley()
assert expectations["aspect_west_side"] == pytest.approx(91.91, abs=0.01)
```

## Files

| File | Role |
|------|------|
| `data/synthetic_valley/generate_synthetic_dem.py` | Generator script with CLI |
| `data/synthetic_valley/README.md` | This file |
| `data/synthetic_valley/synthetic_valley_utm.tif` | Generated output |
| `data/dem.tif` | Existing geographic CRS test DEM |
