#!/usr/bin/env python3
"""
Topology-Preserving Smoothing Algorithm

Implementation of the Alternate Sequential Filter controlled by Topology (ASFT)
based on the paper:
[CB04] M. Couprie and G. Bertrand:
"Topology preserving alternating sequential filter for smoothing 2D and 3D objects"
Journal of Electronic Imaging, Vol. 13, No. 4, pp. 720-730, 2004.

This algorithm smooths binary images while preserving their topology.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import argparse
from PIL import Image

try:
    from numba import jit
    from numba.typed import List as NumbaList
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# TOPOLOGICAL NUMBER LOOKUP TABLE
# ============================================================================
TOPO_TAB = np.array([
    [1,0], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1],
    [1,1], [2,2], [2,2], [2,2], [1,1], [1,1], [1,1], [1,1],
    [1,1], [2,2], [2,2], [2,2], [1,1], [1,1], [1,1], [1,1],
    [1,1], [2,2], [2,2], [2,2], [1,1], [1,1], [1,1], [1,1],
    [1,1], [2,2], [2,2], [2,2], [2,2], [2,2], [2,2], [2,2],
    [2,2], [3,3], [3,3], [3,3], [2,2], [2,2], [2,2], [2,2],
    [1,1], [2,2], [2,2], [2,2], [1,1], [1,1], [1,1], [1,1],
    [1,1], [2,2], [2,2], [2,2], [1,1], [1,1], [1,1], [1,1],
    [1,1], [1,1], [2,2], [1,1], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [3,3], [2,2], [3,3], [2,2],
    [3,3], [3,3], [4,4], [3,3], [3,3], [2,2], [3,3], [2,2],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [1,1], [1,1], [2,2], [1,1], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [2,2], [1,1], [2,2], [1,1],
    [2,2], [2,2], [3,3], [2,2], [2,2], [1,1], [2,2], [1,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
    [1,1], [1,1], [2,2], [1,1], [1,1], [0,1], [1,1], [0,1],
], dtype=np.int32)

# 8-neighbor offsets: E, NE, N, NW, W, SW, S, SE
NEIGHBOR_DY = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
NEIGHBOR_DX = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)


# ============================================================================
# JIT-COMPILED CORE FUNCTIONS
# ============================================================================

@jit(nopython=True, cache=True)
def _get_mask_ge(img, y, x):
    """Get mask where neighbors >= center value."""
    h, w = img.shape
    val = img[y, x]
    v = 0
    if x + 1 < w and img[y, x + 1] >= val: v |= 1
    if y > 0 and x + 1 < w and img[y - 1, x + 1] >= val: v |= 2
    if y > 0 and img[y - 1, x] >= val: v |= 4
    if y > 0 and x > 0 and img[y - 1, x - 1] >= val: v |= 8
    if x > 0 and img[y, x - 1] >= val: v |= 16
    if y + 1 < h and x > 0 and img[y + 1, x - 1] >= val: v |= 32
    if y + 1 < h and img[y + 1, x] >= val: v |= 64
    if y + 1 < h and x + 1 < w and img[y + 1, x + 1] >= val: v |= 128
    return v


@jit(nopython=True, cache=True)
def _get_mask_le(img, y, x):
    """Get mask where neighbors <= center value."""
    h, w = img.shape
    val = img[y, x]
    v = 0
    if x + 1 < w and img[y, x + 1] <= val: v |= 1
    if y > 0 and x + 1 < w and img[y - 1, x + 1] <= val: v |= 2
    if y > 0 and img[y - 1, x] <= val: v |= 4
    if y > 0 and x > 0 and img[y - 1, x - 1] <= val: v |= 8
    if x > 0 and img[y, x - 1] <= val: v |= 16
    if y + 1 < h and x > 0 and img[y + 1, x - 1] <= val: v |= 32
    if y + 1 < h and img[y + 1, x] <= val: v |= 64
    if y + 1 < h and x + 1 < w and img[y + 1, x + 1] <= val: v |= 128
    return v


@jit(nopython=True, cache=True)
def _get_mask_lt(img, y, x):
    """Get mask where neighbors < center value."""
    h, w = img.shape
    val = img[y, x]
    v = 0
    if x + 1 < w and img[y, x + 1] < val: v |= 1
    if y > 0 and x + 1 < w and img[y - 1, x + 1] < val: v |= 2
    if y > 0 and img[y - 1, x] < val: v |= 4
    if y > 0 and x > 0 and img[y - 1, x - 1] < val: v |= 8
    if x > 0 and img[y, x - 1] < val: v |= 16
    if y + 1 < h and x > 0 and img[y + 1, x - 1] < val: v |= 32
    if y + 1 < h and img[y + 1, x] < val: v |= 64
    if y + 1 < h and x + 1 < w and img[y + 1, x + 1] < val: v |= 128
    return v


@jit(nopython=True, cache=True)
def _get_mask_gt(img, y, x):
    """Get mask where neighbors > center value."""
    h, w = img.shape
    val = img[y, x]
    v = 0
    if x + 1 < w and img[y, x + 1] > val: v |= 1
    if y > 0 and x + 1 < w and img[y - 1, x + 1] > val: v |= 2
    if y > 0 and img[y - 1, x] > val: v |= 4
    if y > 0 and x > 0 and img[y - 1, x - 1] > val: v |= 8
    if x > 0 and img[y, x - 1] > val: v |= 16
    if y + 1 < h and x > 0 and img[y + 1, x - 1] > val: v |= 32
    if y + 1 < h and img[y + 1, x] > val: v |= 64
    if y + 1 < h and x + 1 < w and img[y + 1, x + 1] > val: v |= 128
    return v


@jit(nopython=True, cache=True)
def _pdestr4(img, topo_tab, y, x):
    """Test if point is destructible - 4-connected minima."""
    h, w = img.shape
    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return False
    t_ge = _get_mask_ge(img, y, x)
    return topo_tab[t_ge, 0] == 1 and topo_tab[t_ge, 1] == 1


@jit(nopython=True, cache=True)
def _pdestr8(img, topo_tab, y, x):
    """Test if point is destructible - 8-connected minima."""
    h, w = img.shape
    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return False
    t_lt = _get_mask_lt(img, y, x)
    return topo_tab[t_lt, 0] == 1 and topo_tab[t_lt, 1] == 1


@jit(nopython=True, cache=True)
def _pconstr4(img, topo_tab, y, x):
    """Test if point is constructible - 4-connected minima."""
    h, w = img.shape
    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return False
    t_gt = _get_mask_gt(img, y, x)
    return topo_tab[t_gt, 0] == 1 and topo_tab[t_gt, 1] == 1


@jit(nopython=True, cache=True)
def _pconstr8(img, topo_tab, y, x):
    """Test if point is constructible - 8-connected minima."""
    h, w = img.shape
    if y <= 0 or y >= h - 1 or x <= 0 or x >= w - 1:
        return False
    t_le = _get_mask_le(img, y, x)
    return topo_tab[t_le, 0] == 1 and topo_tab[t_le, 1] == 1


@jit(nopython=True, cache=True)
def _alpha8m(img, y, x):
    """Return max of neighbors strictly less than img[y,x]."""
    h, w = img.shape
    val = img[y, x]
    alpha = -1
    for k in range(8):
        ny = y + NEIGHBOR_DY[k]
        nx = x + NEIGHBOR_DX[k]
        if 0 <= ny < h and 0 <= nx < w:
            v = img[ny, nx]
            if v < val and v > alpha:
                alpha = v
    return val if alpha < 0 else alpha


@jit(nopython=True, cache=True)
def _alpha8p(img, y, x):
    """Return min of neighbors strictly greater than img[y,x]."""
    h, w = img.shape
    val = img[y, x]
    alpha = 256
    for k in range(8):
        ny = y + NEIGHBOR_DY[k]
        nx = x + NEIGHBOR_DX[k]
        if 0 <= ny < h and 0 <= nx < w:
            v = img[ny, nx]
            if v > val and v < alpha:
                alpha = v
    return val if alpha > 255 else alpha


@jit(nopython=True, cache=True)
def _delta4m(img, topo_tab, y, x):
    """Compute delta- for 4-connected minima."""
    h, w = img.shape
    val = img[y, x]
    # Work on local copy
    while True:
        t_ge = _get_mask_ge(img, y, x)
        if not (topo_tab[t_ge, 0] == 1 and topo_tab[t_ge, 1] == 1):
            break
        new_val = _alpha8m(img, y, x)
        if new_val >= val:
            break
        img[y, x] = new_val
        val = new_val
    result = img[y, x]
    return result


@jit(nopython=True, cache=True)
def _delta8m(img, topo_tab, y, x):
    """Compute delta- for 8-connected minima."""
    h, w = img.shape
    val = img[y, x]
    while True:
        t_lt = _get_mask_lt(img, y, x)
        if not (topo_tab[t_lt, 0] == 1 and topo_tab[t_lt, 1] == 1):
            break
        new_val = _alpha8m(img, y, x)
        if new_val >= val:
            break
        img[y, x] = new_val
        val = new_val
    result = img[y, x]
    return result


@jit(nopython=True, cache=True)
def _delta4p(img, topo_tab, y, x):
    """Compute delta+ for 4-connected minima."""
    h, w = img.shape
    val = img[y, x]
    while True:
        t_gt = _get_mask_gt(img, y, x)
        if not (topo_tab[t_gt, 0] == 1 and topo_tab[t_gt, 1] == 1):
            break
        new_val = _alpha8p(img, y, x)
        if new_val <= val:
            break
        img[y, x] = new_val
        val = new_val
    result = img[y, x]
    return result


@jit(nopython=True, cache=True)
def _delta8p(img, topo_tab, y, x):
    """Compute delta+ for 8-connected minima."""
    h, w = img.shape
    val = img[y, x]
    while True:
        t_le = _get_mask_le(img, y, x)
        if not (topo_tab[t_le, 0] == 1 and topo_tab[t_le, 1] == 1):
            break
        new_val = _alpha8p(img, y, x)
        if new_val <= val:
            break
        img[y, x] = new_val
        val = new_val
    result = img[y, x]
    return result


@jit(nopython=True, cache=True)
def _compute_delta_m(img_copy, topo_tab, y, x, use_4conn):
    """Compute delta- using a local copy for computation."""
    h, w = img_copy.shape
    orig_val = img_copy[y, x]
    
    if use_4conn:
        result = _delta4m(img_copy, topo_tab, y, x)
    else:
        result = _delta8m(img_copy, topo_tab, y, x)
    
    # Restore original value
    img_copy[y, x] = orig_val
    return result


@jit(nopython=True, cache=True)
def _compute_delta_p(img_copy, topo_tab, y, x, use_4conn):
    """Compute delta+ using a local copy for computation."""
    h, w = img_copy.shape
    orig_val = img_copy[y, x]
    
    if use_4conn:
        result = _delta4p(img_copy, topo_tab, y, x)
    else:
        result = _delta8p(img_copy, topo_tab, y, x)
    
    # Restore original value
    img_copy[y, x] = orig_val
    return result


# ============================================================================
# HOMOTOPIC THINNING - Faithful to C implementation
# ============================================================================

@jit(nopython=True, cache=True)
def _lhthindelta(F, G, topo_tab, connex):
    """
    Homotopic thinning: lower F toward G while preserving topology.
    Faithful implementation of lhthindelta from C code.
    
    Uses LIFO (stack) based processing with stored delta values.
    """
    h, w = F.shape
    N = h * w
    
    # For topology, connectivity is swapped for minima
    use_4conn = (connex == 8)
    
    # Use arrays as stacks (with top pointer)
    # Stack stores encoded values: (pixel_index << 8) | delta
    stack1 = np.zeros(N, dtype=np.int64)
    stack2 = np.zeros(N, dtype=np.int64)
    top1 = 0
    top2 = 0
    
    # Track which pixels are in stack
    in_stack = np.zeros(N, dtype=np.uint8)
    
    # Initialize stack with all destructible points
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            p = y * w + x
            if G[y, x] < F[y, x]:
                if use_4conn:
                    is_destr = _pdestr4(F, topo_tab, y, x)
                else:
                    is_destr = _pdestr8(F, topo_tab, y, x)
                
                if is_destr:
                    delta = _compute_delta_m(F, topo_tab, y, x, use_4conn)
                    # Encode: pixel index in upper bits, delta in lower 8 bits
                    encoded = (p << 8) | (delta & 0xFF)
                    stack1[top1] = encoded
                    top1 += 1
                    in_stack[p] = 1
    
    # Iterate until convergence
    niter = 0
    max_iter = N * 10
    
    while top1 > 0 and niter < max_iter:
        niter += 1
        
        # First half: lower destructible points
        while top1 > 0:
            top1 -= 1
            encoded = stack1[top1]
            p = int(encoded >> 8)
            a = int(encoded & 0xFF)  # Stored delta from push time
            
            y = p // w
            x = p % w
            in_stack[p] = 0
            
            if use_4conn:
                is_destr = _pdestr4(F, topo_tab, y, x)
            else:
                is_destr = _pdestr8(F, topo_tab, y, x)
            
            if is_destr:
                # Compute current delta
                delta = _compute_delta_m(F, topo_tab, y, x, use_4conn)
                # New value is max of current delta and stored delta
                new_val = max(delta, a)
                # Also constrained by G
                new_val = max(new_val, G[y, x])
                
                if new_val < F[y, x]:
                    F[y, x] = new_val
                    stack2[top2] = p
                    top2 += 1
        
        # Second half: check neighbors and push destructible ones
        while top2 > 0:
            top2 -= 1
            p = int(stack2[top2])
            y = p // w
            x = p % w
            
            # Check self and all 8 neighbors
            for k in range(-1, 9):
                if k == -1:
                    ny, nx = y, x
                else:
                    ny = y + NEIGHBOR_DY[k % 8]
                    nx = x + NEIGHBOR_DX[k % 8]
                
                if ny <= 0 or ny >= h - 1 or nx <= 0 or nx >= w - 1:
                    continue
                
                np_idx = ny * w + nx
                
                if in_stack[np_idx] == 0 and G[ny, nx] < F[ny, nx]:
                    if use_4conn:
                        is_destr = _pdestr4(F, topo_tab, ny, nx)
                    else:
                        is_destr = _pdestr8(F, topo_tab, ny, nx)
                    
                    if is_destr:
                        delta = _compute_delta_m(F, topo_tab, ny, nx, use_4conn)
                        encoded = (np_idx << 8) | (delta & 0xFF)
                        stack1[top1] = encoded
                        top1 += 1
                        in_stack[np_idx] = 1
    
    return F


@jit(nopython=True, cache=True)
def _lhthickdelta(F, G, topo_tab, connex):
    """
    Homotopic thickening: raise F toward G while preserving topology.
    Faithful implementation of lhthickdelta from C code.
    """
    h, w = F.shape
    N = h * w
    
    use_4conn = (connex == 8)
    
    stack1 = np.zeros(N, dtype=np.int64)
    stack2 = np.zeros(N, dtype=np.int64)
    top1 = 0
    top2 = 0
    in_stack = np.zeros(N, dtype=np.uint8)
    
    # Initialize with constructible points
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            p = y * w + x
            if G[y, x] > F[y, x]:
                if use_4conn:
                    is_constr = _pconstr4(F, topo_tab, y, x)
                else:
                    is_constr = _pconstr8(F, topo_tab, y, x)
                
                if is_constr:
                    delta = _compute_delta_p(F, topo_tab, y, x, use_4conn)
                    encoded = (p << 8) | (delta & 0xFF)
                    stack1[top1] = encoded
                    top1 += 1
                    in_stack[p] = 1
    
    niter = 0
    max_iter = N * 10
    
    while top1 > 0 and niter < max_iter:
        niter += 1
        
        # First half: raise constructible points
        while top1 > 0:
            top1 -= 1
            encoded = stack1[top1]
            p = int(encoded >> 8)
            a = int(encoded & 0xFF)
            
            y = p // w
            x = p % w
            in_stack[p] = 0
            
            if use_4conn:
                is_constr = _pconstr4(F, topo_tab, y, x)
            else:
                is_constr = _pconstr8(F, topo_tab, y, x)
            
            if is_constr:
                delta = _compute_delta_p(F, topo_tab, y, x, use_4conn)
                # New value is min of current delta and stored delta
                new_val = min(delta, a)
                # Also constrained by G
                new_val = min(new_val, G[y, x])
                
                if new_val > F[y, x]:
                    F[y, x] = new_val
                    stack2[top2] = p
                    top2 += 1
        
        # Second half: check neighbors
        while top2 > 0:
            top2 -= 1
            p = int(stack2[top2])
            y = p // w
            x = p % w
            
            for k in range(-1, 9):
                if k == -1:
                    ny, nx = y, x
                else:
                    ny = y + NEIGHBOR_DY[k % 8]
                    nx = x + NEIGHBOR_DX[k % 8]
                
                if ny <= 0 or ny >= h - 1 or nx <= 0 or nx >= w - 1:
                    continue
                
                np_idx = ny * w + nx
                
                if in_stack[np_idx] == 0 and G[ny, nx] > F[ny, nx]:
                    if use_4conn:
                        is_constr = _pconstr4(F, topo_tab, ny, nx)
                    else:
                        is_constr = _pconstr8(F, topo_tab, ny, nx)
                    
                    if is_constr:
                        delta = _compute_delta_p(F, topo_tab, ny, nx, use_4conn)
                        encoded = (np_idx << 8) | (delta & 0xFF)
                        stack1[top1] = encoded
                        top1 += 1
                        in_stack[np_idx] = 1
    
    return F


def homotopic_thinning(img: np.ndarray, target: np.ndarray, connex: int = 8) -> np.ndarray:
    """Homotopic thinning wrapper."""
    F = img.copy().astype(np.int32)
    G = target.astype(np.int32)
    return _lhthindelta(F, G, TOPO_TAB, connex).astype(np.uint8)


def homotopic_thickening(img: np.ndarray, target: np.ndarray, connex: int = 8) -> np.ndarray:
    """Homotopic thickening wrapper."""
    F = img.copy().astype(np.int32)
    G = target.astype(np.int32)
    return _lhthickdelta(F, G, TOPO_TAB, connex).astype(np.uint8)


# ============================================================================
# MORPHOLOGICAL OPERATIONS USING DISTANCE TRANSFORM
# ============================================================================

def binary_dilate_disk(img: np.ndarray, radius: int) -> np.ndarray:
    """Dilate binary image with disk SE using distance transform."""
    # Distance from background pixels
    dist = ndimage.distance_transform_edt(img == 0)
    dilated = (dist <= radius).astype(np.uint8) * 255
    return dilated


def binary_erode_disk(img: np.ndarray, radius: int) -> np.ndarray:
    """Erode binary image with disk SE using distance transform."""
    # Distance from foreground pixels to nearest background
    dist = ndimage.distance_transform_edt(img > 0)
    eroded = (dist > radius).astype(np.uint8) * 255
    return eroded


# ============================================================================
# HOMOTOPIC PSEUDO-OPENING AND PSEUDO-CLOSING (matching C implementation)
# ============================================================================

def hp_closing_disk(img: np.ndarray, radius: int, connex: int = 8) -> np.ndarray:
    """
    Homotopic pseudo-closing with disk structuring element.
    Matches hpclosingdisc from C code.
    """
    img_sav = img.copy()
    
    # Step 1: Dilate
    dilated = binary_dilate_disk(img, radius)
    
    # Step 2: Homotopic thickening toward dilated
    result = homotopic_thickening(img, dilated, connex)
    
    # Step 3: Erode the thickened result
    eroded = binary_erode_disk(result, radius)
    
    # Step 4: max(eroded, original)
    constraint = np.maximum(eroded, img_sav)
    
    # Step 5: Homotopic thinning toward constraint
    result = homotopic_thinning(result, constraint, connex)
    
    return result


def hp_opening_disk(img: np.ndarray, radius: int, connex: int = 8) -> np.ndarray:
    """
    Homotopic pseudo-opening with disk structuring element.
    Matches hpopeningdisc from C code.
    """
    img_sav = img.copy()
    
    # Step 1: Erode
    eroded = binary_erode_disk(img, radius)
    
    # Step 2: Homotopic thinning toward eroded
    result = homotopic_thinning(img, eroded, connex)
    
    # Step 3: Dilate the thinned result
    dilated = binary_dilate_disk(result, radius)
    
    # Step 4: min(dilated, original)
    constraint = np.minimum(dilated, img_sav)
    
    # Step 5: Homotopic thickening toward constraint
    result = homotopic_thickening(result, constraint, connex)
    
    return result


# ============================================================================
# CONDITIONAL HOMOTOPIC PSEUDO-OPENING AND PSEUDO-CLOSING
# ============================================================================

def cond_hp_closing_disk(img: np.ndarray, cond: np.ndarray, radius: int, connex: int = 8) -> np.ndarray:
    """
    Conditional homotopic pseudo-closing with disk structuring element.
    The constraint 'cond' limits dilation - prevents expanding into constraint region.
    Matches condhpclosingdisc from C code.
    
    cond should be the medial axis of the BACKGROUND (inverse image).
    """
    img_sav = img.copy()
    
    # Step 1: Dilate
    dilated = binary_dilate_disk(img, radius)
    
    # Step 2: Subtract constraint - where cond is set, clear the dilation target
    # In C: for (i = 0; i < N; i++) if (C[i]) T[i] = 0;
    dilated[cond > 0] = 0
    
    # Step 3: Homotopic thickening toward dilated
    result = homotopic_thickening(img, dilated, connex)
    
    # Step 4: Erode the thickened result
    eroded = binary_erode_disk(result, radius)
    
    # Step 5: max(eroded, original)
    constraint = np.maximum(eroded, img_sav)
    
    # Step 6: Homotopic thinning toward constraint
    result = homotopic_thinning(result, constraint, connex)
    
    return result


def cond_hp_opening_disk(img: np.ndarray, cond: np.ndarray, radius: int, connex: int = 8) -> np.ndarray:
    """
    Conditional homotopic pseudo-opening with disk structuring element.
    The constraint 'cond' limits erosion - prevents shrinking below constraint.
    Matches condhpopeningdisc from C code.
    
    cond should be the medial axis of the FOREGROUND (original image).
    """
    img_sav = img.copy()
    
    # Step 1: Erode
    eroded = binary_erode_disk(img, radius)
    
    # Step 2: max(eroded, constraint) - prevent erosion from going below medial axis
    # In C: for (i = 0; i < N; i++) T[i] = max(T[i],C[i]);
    eroded = np.maximum(eroded, cond)
    
    # Step 3: Homotopic thinning toward eroded
    result = homotopic_thinning(img, eroded, connex)
    
    # Step 4: Dilate the thinned result
    dilated = binary_dilate_disk(result, radius)
    
    # Step 5: min(dilated, original)
    constraint = np.minimum(dilated, img_sav)
    
    # Step 6: Homotopic thickening toward constraint
    result = homotopic_thickening(result, constraint, connex)
    
    return result


# ============================================================================
# MEDIAL AXIS COMPUTATION (Meyer's method)
# ============================================================================

def medial_axis_meyer(img: np.ndarray) -> np.ndarray:
    """
    Compute medial axis using Meyer's method (approximation).
    
    1. Compute Euclidean distance transform
    2. Apply unit erosion then unit dilation to distance map
    3. Medial axis points are where eroded-dilated < original distance
    
    Returns binary image with medial axis points set to 255.
    """
    # Distance from background (0) to foreground (255)
    # Note: we want distance FROM background TO foreground pixels
    fg = (img > 0).astype(np.uint8)
    bg = 1 - fg
    
    # Distance transform: distance of each foreground pixel to nearest background
    dist = ndimage.distance_transform_edt(fg)
    
    # Unit erosion (3x3 min filter)
    eroded = ndimage.minimum_filter(dist, size=3)
    
    # Unit dilation (3x3 max filter)
    dilated = ndimage.maximum_filter(eroded, size=3)
    
    # Medial axis: points where opening < original distance
    medax = np.zeros_like(img)
    medax[(fg > 0) & (dilated < dist)] = 255
    
    return medax


# ============================================================================
# ASFT - MAIN ALGORITHM
# ============================================================================

def asft(img: np.ndarray, connex: int = 8, rmax: int = 5, 
         imagec: np.ndarray = None, imagecc: np.ndarray = None) -> np.ndarray:
    """
    Alternate Sequential Filter controlled by Topology.
    Closing then opening for each radius from 1 to rmax.
    
    If imagec and imagecc are provided, uses conditional version:
    - imagec: constraint for object (medial axis of foreground)
    - imagecc: constraint for background (medial axis of inverse)
    """
    result = img.copy()
    
    for radius in range(1, rmax + 1):
        print(f"  ASFT: radius = {radius}/{rmax}")
        if imagec is not None and imagecc is not None:
            result = cond_hp_closing_disk(result, imagecc, radius, connex)
            result = cond_hp_opening_disk(result, imagec, radius, connex)
        else:
            result = hp_closing_disk(result, radius, connex)
            result = hp_opening_disk(result, radius, connex)
    
    return result


def asftmed(img: np.ndarray, connex: int = 8, rmax: int = 5) -> np.ndarray:
    """
    ASFT with medial axis constraints (lasftmed algorithm).
    
    Computes medial axes of both foreground and background, then uses
    them as constraints during ASFT to preserve thin features.
    """
    print("  Computing medial axis of foreground...")
    medaxis = medial_axis_meyer(img)
    
    print("  Computing medial axis of background...")
    # Invert image for background medial axis
    inv_img = 255 - img
    medaxis_inv = medial_axis_meyer(inv_img)
    
    print("  Running constrained ASFT...")
    return asft(img, connex, rmax, medaxis, medaxis_inv)


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def scale_binary_image(img: np.ndarray, scale: int) -> np.ndarray:
    """Scale a binary image by integer factor."""
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def load_image(filepath: str) -> np.ndarray:
    """Load an image from any supported format and return as numpy array.
    
    Supported formats: PNG, JPEG, PGM, BMP, TIFF, GIF, and more.
    The format is automatically detected from the file contents.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Image as numpy array (grayscale or RGB)
    """
    img = Image.open(filepath)
    # Convert to RGB if palette mode, grayscale otherwise
    if img.mode == 'P':
        img = img.convert('RGB')
    elif img.mode == 'RGBA':
        img = img.convert('RGB')
    return np.array(img)


def save_image(img: np.ndarray, filepath: str) -> None:
    """Save an image to any supported format.
    
    Supported formats: PNG, JPEG, PGM, BMP, TIFF, and more.
    The format is automatically determined from the file extension.
    
    Args:
        img: Image as numpy array (uint8)
        filepath: Output path with extension determining format
    """
    pil_img = Image.fromarray(img)
    
    # Convert to appropriate mode based on format
    ext = filepath.lower().split('.')[-1]
    if ext in ('jpg', 'jpeg'):
        # JPEG doesn't support palette or RGBA
        if pil_img.mode not in ('RGB', 'L'):
            pil_img = pil_img.convert('L')
    
    pil_img.save(filepath)


def binarize_image(img: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    """Binarize an image using threshold.
    
    Args:
        img: Input image (grayscale or RGB)
        threshold: Binarization threshold (default: mean intensity)
        
    Returns:
        Binary image (0 or 255)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    if threshold is None:
        threshold = np.mean(img)
    return (img > threshold).astype(np.uint8) * 255


def topology_preserving_smooth(
    img: np.ndarray,
    scale: int = 4,
    smooth_radius: int = 5,
    connex: int = 8,
    threshold: Optional[float] = None,
    use_medial: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Main function: topology-preserving smoothing.
    
    Args:
        img: Input image
        scale: Scaling factor for the image
        smooth_radius: Maximum radius for ASFT
        connex: Connectivity (4 or 8)
        threshold: Binarization threshold (default: mean intensity)
        use_medial: If True, use medial axis constraints (asftmed)
    """
    binary = binarize_image(img, threshold)
    
    print(f"Scaling image by factor {scale}...")
    scaled = scale_binary_image(binary, scale)
    print(f"  Scaled size: {scaled.shape}")
    
    if use_medial:
        print(f"Applying ASFT-MED with rmax={smooth_radius}...")
        smoothed = asftmed(scaled, connex, smooth_radius)
    else:
        print(f"Applying ASFT with rmax={smooth_radius}...")
        smoothed = asft(scaled, connex, smooth_radius)
    
    return smoothed, binary


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Topology-Preserving Image Smoothing")
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Output smoothed image file")
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("-r", "--radius", type=int, default=5)
    parser.add_argument("-c", "--connex", type=int, choices=[4, 8], default=8)
    parser.add_argument("-t", "--threshold", type=float, default=None)
    parser.add_argument("--save-binary", type=str, default=None)
    parser.add_argument("--medial", dest="use_medial", action="store_true", default=True,
                        help="Use medial axis constraints (asftmed, default)")
    parser.add_argument("--no-medial", dest="use_medial", action="store_false",
                        help="Don't use medial axis constraints (plain asft)")
    
    args = parser.parse_args()
    
    if not HAS_NUMBA:
        print("WARNING: numba not installed. Algorithm will be SLOW.")
    
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
