# Topology-Preserving Smoothing - Python Implementation

A Python implementation of the Alternate Sequential Filter controlled by Topology (ASFT) algorithm for smoothing binary images while preserving their topology (number of connected components and holes).

## Example: Einstein Image

| Original | Smoothed (ASFT-MED) | Smoothed (ASFT) |
|:--------:|:-------------------:|:---------------:|
| <img src="img/einstein_original.png" height="300"> | <img src="img/einstein_smoothed_r10.png" height="300"> | <img src="img/einstein_smoothed_ASFT_pure.png" height="300"> |

*Smoothed with: `-s 1 -r 3 -c 4 --medial`* for ASFT-MED and with the same parameters for the pure ASFT.

The algorithm smooths jagged boundaries while **preserving topology** - all connected components and holes from the original image are maintained.

## More Examples

### Example 1: Pixelized stains #1
| Input | Output |
|:-----:|:------:|
| <img src="img/input_6.png" width="300"> | <img src="img/output_6_r10.png" width="300"> |

### Example 2: Pixelized stains #2
| Input | Output |
|:-----:|:------:|
| <img src="img/input_5.png" width="300"> | <img src="img/output_5_r10.png" width="300"> |

### Example 3: QR-code (Connectivity Comparison)
| Input | Output (C8) | Output (C4) |
|:-----:|:-----------:|:-----------:|
| <img src="img/input_3.png" width="200"> | <img src="img/output_3.png" width="200"> | <img src="img/output_3c4.png" width="200"> |

### Example 4: Flying heron
| Input | Output |
|:-----:|:------:|
| <img src="img/input_1.png" width="300"> | <img src="img/output_1_r10.png" width="300"> |

### Example 5: Trees
| Input | Output |
|:-----:|:------:|
| <img src="img/input_4.png" width="300"> | <img src="img/output_4.png" width="300"> |


## Reference

Based on the paper:
> M. Couprie and G. Bertrand: "Topology preserving alternating sequential filter for smoothing 2D and 3D objects", Journal of Electronic Imaging, Vol. 13, No. 4, pp. 720-730, 2004.

See also the dedicated web-page:
> [Topological Smoothing by M. Couprie, G. Bertrand](https://perso.esiee.fr/~coupriem/ts/index.html)

## Installation

### From PyPI (when published)

```bash
# Recommended installation (includes numba for 10-100x speedup)
pip install toposmooth[fast]

# If you have issues installing numba (e.g., unsupported platform):
pip install toposmooth
```

### From Source

```bash
# Clone the repository
git clone https://github.com/vyastrebov/toposmooth.git
cd toposmooth

# Recommended: install with numba
pip install -e ".[fast]"

# Without numba (if installation fails):
pip install -e .
```

### Required packages
- numpy
- scipy  
- Pillow
- numba (included with `[fast]`, provides 10-100x speedup)

## Supported Image Formats

The algorithm supports any bitmap format readable by Pillow:
- **PNG** (`.png`) - recommended for lossless output
- **JPEG** (`.jpg`, `.jpeg`) - input only (lossy compression not ideal for binary output)
- **PGM** (`.pgm`) - portable graymap, matches C implementation
- **BMP** (`.bmp`)
- **TIFF** (`.tiff`, `.tif`)
- And many others...

The output format is automatically determined by the file extension.

## Usage

### Command Line

After installation, you can use the `toposmooth` command or run as a Python module:

```bash
# Using the installed command
toposmooth input.png output.png

# Or using Python module syntax
python -m toposmooth input.png output.png

# Match C asftmed command: ./asftmed input.pgm 4 3 output.pgm
toposmooth input.pgm output.pgm -s 1 -r 3 -c 4

# With all options
toposmooth input.jpg output.png \
    -s 1           # Scale factor (default: 4, use 1 to match input dimensions)
    -r 5           # Maximum smoothing radius (default: 5)
    -c 4           # Connectivity for white value (1): 4 (neighbor) or 8 (next-neighbour)
    -t 128         # Binarization threshold (default: mean intensity, if the input is not binary)
    --medial       # Use medial axis constraints - strictly preserves the form (default, recommended)
    --no-medial    # Use plain ASFT without medial axis constraints (removes small features)
    --save-binary binary.png  # Save the binarized input image

# Show version
toposmooth --version
```

### Python API

```python
from toposmooth import topology_preserving_smooth, load_image, asftmed
from PIL import Image
import numpy as np

# Load and process an image
img = load_image('input.png')
smoothed, binary = topology_preserving_smooth(
    img,
    scale=1,           # Scale factor (1 = no scaling, matches C)
    smooth_radius=5,   # Maximum ASFT radius
    connex=4,          # Object connectivity (4 or 8)
    threshold=None,    # Binarization threshold (None = mean)
    use_medial=True    # Use medial axis constraints (recommended)
)

# Save result in any format
Image.fromarray(smoothed).save('output.png')   # PNG
Image.fromarray(smoothed).save('output.jpg')   # JPEG
Image.fromarray(smoothed).save('output.pgm')   # PGM
```

## Parameters

- **scale**: Integer scaling factor. The image is upscaled before smoothing to allow sub-pixel smoothing with respect to original pixel size. Use `scale=1` to match the original image dimensions.

- **smooth_radius (rmax)**: Maximum radius of disk structuring elements. The algorithm applies opening/closing operations with disks of radius 1, 2, ..., rmax. Larger values produce smoother results but take longer (linear complexity with rmax), but the default value rmax=5 produces already very good results.

- **connex**: Object connectivity (4 or 8). For 8-connectivity, diagonal neighbors (white pixels, value=1) are considered connected.

- **threshold**: Binarization threshold. Pixels above this value become foreground (255), others become background (0). If None, uses the mean intensity.

- **use_medial / --medial**: When enabled (default), uses medial axis constraints to preserve thin features, recommended.

## Algorithms

### ASFT-MED (default, `--medial`)
Uses medial axes of both foreground and background as constraints. This prevents:
- Thin features from being eroded away
- Small gaps from being filled in


### Plain ASFT (`--no-medial`)
Standard ASFT without constraints. May lose very thin features.


## How It Works

1. **Binarization**: The input image is converted to binary using the threshold.

2. **Scaling** (optional): The binary image is upscaled by the scale factor.

3. **Medial Axis Computation** (ASFT-MED only):
   - Compute medial axis of foreground (skeleton)
   - Compute medial axis of background (inverse skeleton)

4. **ASFT**: For each radius r from 1 to rmax:
   - Apply homotopic pseudo-closing (fills small holes while preserving topology)
   - Apply homotopic pseudo-opening (removes small protrusions while preserving topology)
   - With medial axis constraints, these operations are bounded to preserve thin structures

5. The result is a smoothed binary image with the same topology as the input.

## Performance

- **With numba** (`pip install toposmooth[fast]`): ~1-2 seconds for a 520×372 image with scale=1, radius=3
- **Without numba**: Much slower (minutes instead of seconds) - only use if numba won't install
- The algorithm scales roughly as O(scale² × rmax × width × height)
- First run with numba may take a few seconds for JIT compilation

## Validation

The Python implementation produces **pixel-perfect results** matching the [C implementation](https://perso.esiee.fr/~coupriem/ts/TS_programs.html) for Einstein's binary portrait.

```bash
# C version
./asftmed test/einstein.pgm 4 3 c_output.pgm

# Python version
toposmooth test/einstein.pgm py_output.pgm -s 1 -r 3 -c 4
```

## Test Shapes

Run `python test_shapes.py` to generate comparison panels for various test shapes:

1. **Rectangle** - Grid-aligned rectangle
2. **Square** - Grid-aligned square  
3. **L-Shape** - Grid-aligned L shape
4. **Squares Edge Touch** - Two squares sharing a full edge (1 component for both connectivities)
5. **Squares Single Point Touch** - Two squares touching at exactly one 4-neighbor pixel (1 component for both)
6. **Squares Diagonal Touch** - Two squares touching only at diagonal (2 components for 4-conn, 1 for 8-conn)
7. **Rotated Square 30°** - Square rotated by 30 degrees
8. **Rotated Square 45°** - Square rotated by 45 degrees

Each test shows the original shape and results for:
- Connectivity 4 and 8
- ASFT and ASFT-MED algorithms
- Scale factors 1 and 4

Results are saved to `test_results/` folder, including `all_tests_overview.png` with all shapes combined.

### Sample Test Result

![Test Overview](test_results/all_tests_overview.png)

### Information

+ **Authors:** Claude Opus 4.5 in Cursor environment (mainly), Vladislav A. Yastrebov (marginally)
+ **Validator:** Vladislav A. Yastrebov (CNRS, Mines Paris)
+ **License:** GPL-2.0
+ **Date:** Nov-Dec 2025

