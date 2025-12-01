#!/usr/bin/env python3
"""
Command-line interface for topology-preserving smoothing.

This module allows the package to be run as:
    python -m toposmooth input.png output.png [options]
"""

import argparse
import sys

from . import (
    HAS_NUMBA,
    load_image,
    save_image,
    topology_preserving_smooth,
)


def main():
    parser = argparse.ArgumentParser(
        prog="toposmooth",
        description="Topology-Preserving Image Smoothing using ASFT algorithm"
    )
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Output smoothed image file")
    parser.add_argument("-s", "--scale", type=int, default=4,
                        help="Scale factor (default: 4)")
    parser.add_argument("-r", "--radius", type=int, default=5,
                        help="Maximum smoothing radius (default: 5)")
    parser.add_argument("-c", "--connex", type=int, choices=[4, 8], default=8,
                        help="Connectivity: 4 or 8 (default: 8)")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Binarization threshold (default: mean intensity)")
    parser.add_argument("--save-binary", type=str, default=None,
                        help="Save binarized input image to this path")
    parser.add_argument("--medial", dest="use_medial", action="store_true", default=True,
                        help="Use medial axis constraints (asftmed, default)")
    parser.add_argument("--no-medial", dest="use_medial", action="store_false",
                        help="Don't use medial axis constraints (plain asft)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__import__('toposmooth').__version__}")
    
    args = parser.parse_args()
    
    if not HAS_NUMBA:
        print("WARNING: numba not installed. Performance will be very slow.", file=sys.stderr)
        print("         Install with: pip install toposmooth[fast]", file=sys.stderr)
    
    print(f"Loading {args.input}...")
    img = load_image(args.input)
    print(f"  Image shape: {img.shape}")
    
    smoothed, binary = topology_preserving_smooth(
        img, args.scale, args.radius, args.connex, args.threshold, args.use_medial
    )
    
    print(f"Saving smoothed image to {args.output}...")
    save_image(smoothed, args.output)
    
    if args.save_binary:
        save_image(binary, args.save_binary)
    
    print("Done!")


if __name__ == "__main__":
    main()

