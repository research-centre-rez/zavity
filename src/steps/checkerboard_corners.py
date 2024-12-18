import logging
import traceback

import cv2
import numpy as np
from sklearn.preprocessing import minmax_scale

logger = logging.getLogger(__name__)


def harris(img, radius=2):
    """
    Harris-Stephens corner detector.
    :param img : numpy array containing a single-channel (grayscale) image.
    :param radius : half-width and half-height of the square convolution filters.
    :returns : filtered image as a float32 array of the same shape as the image.
    """
    fimg = img.astype(np.float32) / 255.0
    ix = cv2.Scharr(fimg, cv2.CV_32F, 1, 0)
    iy = cv2.Scharr(fimg, cv2.CV_32F, 0, 1)
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
    kwid = 2 * radius + 1
    kshape = (kwid, kwid)
    m11 = cv2.blur(ix2, kshape)
    m22 = cv2.blur(iy2, kshape)
    m12 = cv2.blur(ixy, kshape)
    tr = m11 + m22
    det = m11 * m22 - m12 * m12
    eps = 1.0e-6
    h = det / (tr + eps)
    return h


def local_maxima(img, radius=3, num=100):
    """
    Locate the most salient local maxima of an image, using a kernel of given size
    :param img : numpy array storing a single-channel image.
    :param radius : half-width of the applied morphological dilation kernel.
    :param num : maximum number of most salient local maxima to return.
    :returns list of at most num (x, y) tuples containing the image coordinates of the found maxima.
    """
    # Dilate and match image and dilated version, getting _all_ the local maxima
    sel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dil=cv2.dilate(img, sel)
    mx = np.where(img == dil, np.ones(img.shape, dtype=np.int32), np.zeros(img.shape, dtype=np.int32))
    h, w = mx.shape
    lm = []
    for y in range(h):
        for x in range(w):
            if mx[y, x] == 1:
                lm.append((x, y))
    # Sort the local maxima by descending image intensity.
    lm = sorted(lm, key=lambda p: img[p[1], p[0]], reverse=True)
    # Traverse the sorted list, suppress a window of the same radius about each maximum, and
    # select the first "num" that have not been suppressed.
    lms=[]
    for x, y in lm:
        if mx[y, x] == 0:
            # It has already been suppressed, discard it.
            continue
        lms.append((x, y))
        if len(lms) == num:
            break
        # Suppress neighbors to avoid duplicating nearby maxima.
        xmin=max(0, x - radius)
        xmax=min(w, x + radius + 1)
        ymin=max(0, y - radius)
        ymax=min(h, y + radius + 1)
        mx[ymin:ymax, xmin:xmax] = 0
    return lms


def voronoi_facets(wid, hei, points):
    """
    Construct the Voronoi diagram of a set of points/
    :param wid, hei : height and width of the diagram
    :param points: iterable of (x, y) tuples
    :returns list of facets of the Voronoi diagram, each facet a list of vertices.
    """
    rect = (0, 0, wid, hei)
    sub2d = cv2.Subdiv2D(rect)
    sub2d.insert(points)
    facet_list, _ = sub2d.getVoronoiFacetList([])
    return facet_list


def voronoi_diagram(wid, hei, facet_list):
    """
    Draws an image of the Voronoi diagram, given its facets.
    :param wid, hei : height and width of the diagram.
    :facet_list: list of facets of the Voronoi diagram, each facet a list of vertices.
    :return numpy array of shape (hei, wid, 3), of type int32, whose RGB values encode indices in the facet_list.
    """
    assert len(facet_list) < (1 << 24), 'Can color only up to 2**24 -1 facets'
    int_facet_list = [np.round(np.array(f)).astype(np.int32) for f in facet_list]
    vdiagram = np.zeros((hei, wid, 3), dtype=np.uint8)
    for i, f in enumerate(int_facet_list):
        color = (i & 255, (i >> 8) & 255, (i >> 16) & 255)
        cv2.fillConvexPoly(vdiagram, np.array([f]), color)
    return vdiagram


def compute_facet_medians(img, vdiagram, num_facets):
    """
    Computes the medians of the image histogram separately in each Voronoi facet.
    :param img : numpy array storing a single-channel image.
    :param vdiagram : numpy array of the same shape as width & height as img, as returned by method voronoi_diagram.
    :param num_facets : number of facets in the Voronoi diagram.
    :returns grayscale image of the same shape as img, with the Voronoi facets colored with the median gray level
             of img within them.
    """
    hei, wid = img.shape
    # Partition the image pixel values into sets, one for each facet
    vsets = [[] for _ in range(num_facets)]
    for y in range(hei):
        for x in range(wid):
            r, g, b = vdiagram[y, x, :]
            i = int(r) | (int(g) << 8) | (int(b) << 8)
            v = img[y, x]
            vsets[i].append(int(v))
    # Compute the list of medians of each set.
    meds=[np.uint8(np.median(v)) if v != [] else 0 for v in vsets]
    # Build the image of medians
    medmap = np.zeros(img.shape, dtype=np.uint8)
    for y in range(hei):
        for x in range(wid):
            r, g, b = vdiagram[y, x, :]
            i = int(r) | (int(g) << 8) | (int(b) << 8)
            medmap[y, x] = meds[i]
    return medmap


def draw_corners(img, crs, show_id=False, valid=None):
    """
    Draw corners on top of the image
    :param img : numpy array storing a single-channel image.
    :crs : iterable of (x, y) tuples.
    """
    assert valid is None or len(valid) == len(crs)
    if img.ndim == 2:
        dimg = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        dimg = img.copy()
    for i, (x, y) in enumerate(crs):
        if valid is None or valid[i]:
            xy = (int(np.round(x)), int(np.round(y)))
            dimg = cv2.circle(dimg, xy, 5, (255, 0, 0), 1)
            if show_id:
                dimg = cv2.putText(dimg, str(i), (xy[0] + 5, xy[1] + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
    return dimg


def bw_area_detection(corners_subpix, bw_img, invert_block_radius=5):
    """
    Segments the black and white regions of a given BW image.
    :param corner_subpix : Nx2 array of feature points.
    :param bw_image : thresholded black-white image.
    :param invert_block_radius : radius of the region to be inverted around each feature point.
    :returns  pair of:
        region_image, an rgb image co-dimensional with bw_img, whose pixels encode the index of a segmented region.
        Cr: list such that Cr[i] is 1 iff region i is white, and 0 if it is black.
    """
    hei, wid = bw_img.shape
    # Build a list of quads around each corner
    quads = []
    for c in corners_subpix:
        x, y = tuple(np.int32(c))
        x0 = max(0, x - invert_block_radius)
        x1 = min(wid - 1, x + invert_block_radius)
        y0 = max(0, y - invert_block_radius)
        y1 = min(hei - 1, y + invert_block_radius)
        quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        quads.append(np.array(quad).reshape((4, 1, 2)))

    white_cc_img = bw_img.copy()
    for q in quads:
        cv2.fillConvexPoly(white_cc_img, q, 0)
    nw, white_cc = cv2.connectedComponents(
        white_cc_img, connectivity=4, ltype=cv2.CV_32S)

    ccbimg = 255 - bw_img
    for q in quads:
        cv2.fillConvexPoly(ccbimg, q, 0)
    nb, black_cc = cv2.connectedComponents(
        ccbimg, connectivity=4, ltype=cv2.CV_32S)

    region_image = black_cc
    region_colors = {}
    for i in range(hei):
        for j in range(wid):
            r = region_image[i, j]
            if region_image[i, j] == 0:
                w = white_cc[i, j]
                if w != 0:
                    r = nb + w - 1
                    region_image[i, j] = r
                    region_colors[r] = 1
            else:
                region_colors[r] = 0

    num_cc = nb + nw - 2
    assert len(region_colors) == num_cc
    return region_image, region_colors


def corner_region_association(corners_subpix, region_image, Cr, radius=13):
    """
    Associates corners to black/white regions within a neighborhood of them of given radius.

    """
    hei, wid = region_image.shape
    num_cc = len(Cr)
    F = corners_subpix
    FI = np.int32(F)
    Fr = {}
    rad = 13
    for i, f in enumerate(FI):
        x, y = tuple(f)
        x0 = max(0, x - rad)
        x1 = min(wid - 1, x + rad)
        y0 = max(0, y - rad)
        y1 = min(hei - 1, y + rad)
        regs = set()
        # "Walk" around the feature along a square or radius rad.
        for xx in range(x0, x1):
            for yy in (y0, y1):
                r = region_image[yy, xx]
                if r != 0:
                    regs.add(r)
        for yy in range(y0, y1):
            for xx in (x0, x1):
                r = region_image[yy, xx]
                if r != 0:
                    regs.add(r)
        Fr[i] = regs

    # Invert index and remove background and irregular regions and features.
    Rf = {i: set() for i in range(num_cc + 1)}
    for i in list(Fr.keys()):
        regs = Fr[i]
        if len(regs) != 4:
            Fr.pop(i)
        else:
            num_whites = sum(1 for r in regs if Cr[r] == 1)
            if num_whites != 2:
                Fr.pop(i)
            else:
                for r in regs:
                    Rf[r].add(i)

    for r in list(Rf.keys()):
        if len(Rf[r]) != 4:
            Rf.pop(r)
    for f, rs in list(Fr.items()):
        rs = [x for x in rs if x in Rf]
        Fr[f] = set(rs)

    return Fr, Rf


def draw_regions(img, corners_subpix, Fr, Rf, Cr, show_id=True):
    """
    Draws the regions whose indices are the keys of Rf, in
    yellow if the color in Cr is 1 (white), else cyan.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    for r, fs in Rf.items():
        crs = np.int32([corners_subpix[i] for i in fs])
        crs_reverse = np.array([crs[0,:], crs[2,:], crs[1,:], crs[3,:]])
        color = yellow if Cr[r] == 1 else cyan
        bgr = cv2.fillPoly(bgr, [crs.reshape((-1, 1, 2))], color=color)
        bgr = cv2.fillPoly(bgr, [crs_reverse.reshape((-1, 1, 2))], color=yellow if Cr[r] == 1 else cyan)
        if show_id:
            ctr = tuple(np.int32(np.mean(crs, 0)).tolist())
            bgr = cv2.putText(bgr, str(r), ctr, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0))
    return bgr


def show_regions(img, corners_subpix, Fr, Rf, Cr, show_id=True):
    bgr = draw_regions(img, corners_subpix, Fr, Rf, Cr, show_id=True)
    import matplotlib.pyplot as plt
    plt.imshow(bgr)
    plt.show()


def common_regions_of_corners(i, j, Fr):
    return list(Fr[i].intersection(Fr[j]))

def are_connected_regions(regions, Cr):
    return len(regions) == 2 and Cr[regions[0]] != Cr[regions[1]]

def are_connected_corners(i, j, Fr, Cr):
    regions = common_regions_of_corners(i, j, Fr)
    return are_connected_regions(regions, Cr)

def regions_adjoining_side(i, j, Fr, Cr):
    regions = common_regions_of_corners(i, j, Fr)
    if not are_connected_regions(regions, Cr):
        return None
    else:
        return regions[0], regions[1]


def region_sides(Fr, Rf, Cr):
    Rs = {}
    for r, fs in Rf.items():
        assert len(fs) == 4
        sides = []
        lfs = list(fs)
        for i, f in enumerate(lfs[:3]):
            for g in lfs[(i + 1):]:
                if are_connected_corners(f, g, Fr, Cr):
                    sides.append((f, g))
        if len(sides) <= 4:
            # "This is unknown anomaly, there are extra faces causing neigboring of the points. Probably out of checkerboard."
            Rs[r] = sides
    return Rs


def is_inner_region(r, Rs):
    return r in Rs and len(Rs[r]) == 4


def regions_adjoining_region(r, Rs, Fr, Cr):
    adj = set()
    for s, t in Rs[r]:
        adj.update(regions_adjoining_side(s, t, Fr, Cr))
    return list(adj.difference([r]))


def opposite_side(i, r, Rs):
    """
    Compute the opposite side to a given one in a region.
    :param i : index of one of the sides of region r.
    :param r : index of a region.
    :param Rs : the region-sides table
    :return the index j of the side opposite to the i-th one in Rs[r]
    """
    assert is_inner_region(r, Rs) and 0 <= i < 4
    sides = Rs[r]
    si = set(sides[i])
    for j, side in enumerate(sides):
        if j != i and not set(side).intersection(si):
            return j


def adjoining_sides(i, r, Rs):
    """
    Compute the pair of adjoining sides to a given one in a region.
    :param i : index of one of the sides of region r.
    :param r : index of a region.
    :param Rs : the region-sides table
    :return the indices (j, k) of the sides adjoining to the i-th one in Rs[r]
    """
    j = opposite_side(i, r, Rs)
    addjoining_sides = tuple(set(range(4)).difference([i, j]))
    assert len(addjoining_sides) == 2, "Unexpected topology. There is more or less than two adjoining sides."

    return addjoining_sides


def sort_sides_of_region(r, Rs):
    assert is_inner_region(r, Rs)
    i0 = 0
    i2 = opposite_side(i0, r, Rs)
    i1, i3 = adjoining_sides(i0, r, Rs)
    ss = Rs[r]
    s0, s1, s2, s3 = ss[i0], ss[i1], ss[i2], ss[i3]
    s01 = s0[1]
    if not s01 in s1:
        s1, s3 = s3, s1
        assert s01 in s1
    if s01 != s1[0]:
        s1 = (s1[1], s1[0])
    s11 = s1[1]
    assert s11 in s2
    if s11 != s2[0]:
        s2 = (s2[1], s2[0])
    s21 = s2[1]
    assert s21 in s3
    if s21 != s3[0]:
        s3 = (s3[1], s3[0])
    assert s3[1] == s0[0]
    Rs[r] = [s0, s1, s2, s3]


def sort_sides_consistently(r1, r0, i0, Rs):
    """
    Given region r1, adjacent to region r0 along the i0-th side of r0,
    sorts the sides and corners of r1 in Rs to be consistent with r0.
    """
    assert is_inner_region(r0, Rs)
    assert is_inner_region(r1, Rs)
    assert 0 <= i0 < 4
    found = -1
    c0, c1 = Rs[r0][i0]
    found = False
    for i1, s1 in enumerate(Rs[r1]):
        if c0 in s1 and c1 in s1:
            found = True
            break
    assert found
    s1 = (c1, c0)
    if i1 != 0:
        s0 = Rs[r1][0]
        Rs[r1][0] = s1
        Rs[r1][i1] = s0
    else:
        Rs[r1][0] = s1
    sort_sides_of_region(r1, Rs)
    i_shft = [i % 4 for i in range(i0 + 2, i0 + 2 + 4)]
    ss1 = Rs[r1]
    ss1_shft = [None] * 4
    for i in range(4):
        j = (i0 + i + 2) % 4
        ss1_shft[j] = ss1[i]
    Rs[r1] = ss1_shft


def feature_offsets(corners_subpix, Fr, Rs, Cr):
    """
    Sort the detected corners into a 2D grid.
    :param corners_subpix List of (x, y) image position of detected checkerboard corners.
    :param Fr Corner-to-regions table as returned by function corner_region_association().
    :param Rs Region-to-sides tables as returned by function region_sides().
    :param Cr Region-colors table as returned by function bw_area_detection().
    :return Pair (H, G), where
       H: an NxM numpy array of integers, such that its cell
          at position (i,j) contains the index of a corner in corners_subpix that has been
          located therein. A value of -1 indicates that no corners has been located at (i, j)
       G: a dictionary mapping the indices of elements of corners_subpix into (i, j) positions
          into a 2D grid.
    """
    # Comments in the code will follow Section 3.1.1 of the paper.

    # Declare and fill with false values a boolean table Vs, having
    # the same size as the region sides table Rs . A region t is said to
    # be “entered from side s of the adjoining region r" if Vs[r][s] == True.
    Vs = {r: ([False] * len(sides)) for r, sides in Rs.items()}

    # Declare and fill with false values a boolean list Vr of length
    # equal to Rs. A region r is said to be “visited” when Vr[R] is true.
    Vr = {r: False for r in Rs.keys()}

    # Declare and fill with false values a boolean list I of size equal to Fr.
    # A feature of index k is said to be “inconsistent” if I[k] is true.
    I = {f: False for f in Fr.keys()}

    # Declare a dictionary F, keyed on the corner index, whose values
    # are the found row/column offset of that corner.
    G = {}

    # Declare an empty stack S.
    S = []

    # Start from an inner region r adjoining only inner regions.
    found = False
    innerity = {}
    for r in Rs.keys():
        adj = regions_adjoining_region(r, Rs, Fr, Cr)
        innerity[r] = np.sum([is_inner_region(a, Rs) for a in adj])
        if is_inner_region(r, Rs):
            if all([is_inner_region(a, Rs) for a in adj]):
                found = True
                break
    try:
        if not found:
            if len(innerity) == 0:
                return None
            r, _ = sorted(innerity.items(), key=lambda x: x[1], reverse=True)[0]

        # Topologically sort the sides of region r.
        sort_sides_of_region(r, Rs)
    except AssertionError as a:
        logger.debug(f"Unexpected topology for region {r}")
        return None

    # Initialize traversal: mark region r visited
    Vr[r] = True

    # Initialize the region row and col. increments from r
    o_r = 0
    o_c = 0

    S.append((r, o_r, o_c))

    # Traversal: exterior loop over the stack.
    while S:
        r, o_r, o_c = S.pop()
        # Inner loop
        while True:
            # Search for a side not already crossed from r
            found = False
            for i in range(4):
                if not Vs[r][i]:
                    s = i
                    found = True
                    break
            if not found:
                break  # out of while True

            # Mark r entered from s, compute its corners’ offsets.
            Vs[r][s] = True

            if s == 0:
                r1 = o_r
                c1 = o_c
                r2 = r1
                c2 = c1 + 1
            elif s == 1:
                r1 = o_r
                c1 = o_c + 1
                r2 = r1 + 1
                c2 = c1
            elif s == 2:
                r1 = o_r + 1
                c1 = o_c + 1
                r2 = r1
                c2 = c1 - 1
            else:
                assert s == 3
                r1 = o_r + 1
                c1 = o_c
                r2 = r1 - 1
                c2 = c1

            # Enter offsets in grid if not inconsistent.
            f1, f2 = Rs[r][s]
            if I[f1] or I[f2]:
                logger.debug('inconsistent 1: {} {}'.format(I[f1], I[f2]))
                continue  # next "while True" iteration
            g1 = G.get(f1)
            g2 = G.get(f2)
            if g1 == (r1, c1) and g2 == (r2, c2):
                pass
            elif g1 is None and g2 is None:
                G[f1] = (r1, c1)
                G[f2] = (r2, c2)
            elif g1 == (r1, c1) and g2 is None:
                G[f2] = (r2, c2)
            elif g1 is None and g2 == (r2, c2):
                G[f1] = (r1, c1)
            else:
                I[f1] = I[f2] = True
                if f1 in G:
                    G.pop(f1)
                if f2 in G:
                    G.pop(f2)

            if I[f1] or I[f2]:
                logger.debug('inconsistent 2: {} {}'.format(I[f1], I[f2]))
                continue  # next "while True" iteration

            # Step across side if adjoining region is inner too.
            a, b = regions_adjoining_side(f1, f2, Fr, Cr)
            if a == r:
                r_adj = b
            else:
                r_adj = a

            if not is_inner_region(r_adj, Rs):
                continue  # next "while True" iteration

            # If r_adj hasn’t been visited yet, topologically sort its sides
            # consistently with those of r.
            if not Vr[r_adj]:
                sort_sides_consistently(r_adj, r, s, Rs)
                Vr[r_adj] = True

            Vs[r_adj][(s + 2) % 4] = True
            # Update region offsets
            if s == 0:
                o_r -= 1
            elif s == 1:
                o_c += 1
            elif s == 2:
                o_r += 1
            else:
                assert s == 3
                o_c -= 1

            # New current region
            r = r_adj
            S.append((r, o_r, o_c))

        # end "while True"
    # end while Sr

    # Extract the valid inner rectangle of G
    G_vals = np.array(list(G.values()))
    MM, NN = np.max(G_vals, 0).tolist()
    mm, nn = np.min(G_vals, 0).tolist()
    m = MM - mm + 1
    n = NN - nn + 1
    H = -np.ones((m, n), dtype=np.int32)
    for f, (r, c) in G.items():
        o_r = r - mm
        o_c = c - nn
        H[o_r, o_c] = f
        G[f] = (o_r, o_c)

    # Rotate/Reflect H to conform to the image
    crs = np.array([list(c) for c in corners_subpix])
    dr = np.array([0., 0.])
    dc = np.array([0., 0.])
    for i in range(1, m):
        for j in range(1, n):
            hij = H[i, j]
            if hij >= 0:
                pij = crs[hij, :]
                if H[i - 1, j] >= 0:
                    qij = crs[H[i - 1, j], :]
                    dr += pij - qij
                if H[i, j - 1] >= 0:
                    qij = crs[H[i, j - 1], :]
                    dc += pij - qij

    return G, H


def process_image(frame, frac=4):
    # TODO: all constants are dependent on image resolution ... adapt

    img = cv2.resize(frame, (frame.shape[1] // frac, frame.shape[0] // frac))
    img = minmax_scale(img[:,:,2], feature_range=(0,255)).astype(np.uint8) #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corner_strength = harris(img, radius=1)
    corners = local_maxima(corner_strength, radius=11, num=300)
    corners = np.array(corners)

    corners = np.array([corner for corner in corners if
               np.sqrt((corner[0] - img.shape[1] // 2) ** 2 + (corner[1] - img.shape[0] // 2) ** 2) < 1100 // frac])

    subpix_window = 10
    zero_zone = 1
    num_iterations = 10
    subpix_threshold = 0.03

    corners_subpix = cv2.cornerSubPix(
        img, np.float32(np.array(corners).reshape(-1, 1, 2).tolist()),
        (subpix_window, subpix_window), (zero_zone, zero_zone),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
         num_iterations, subpix_threshold))

    corners_subpix = np.stack([
        np.clip([corner[0] for corner in corners_subpix[:,0,:]], 0, img.shape[1]),
        np.clip([corner[1] for corner in corners_subpix[:,0,:]], 0, img.shape[0])
    ], axis=1).tolist()
    #corners_subpix = (np.reshape(corners_subpix, (-1, 2)).tolist())

    hei, wid = img.shape
    facet_list = voronoi_facets(wid, hei, corners_subpix)

    vdiagram = voronoi_diagram(wid, hei, facet_list)
    # show_image(vdiagram)

    num_corners = len(corners_subpix)
    facet_medians = compute_facet_medians(img, vdiagram, num_corners)
    #with_corners_on_top = draw_corners(facet_medians, corners_subpix)
    #show_image(with_corners_on_top)

    threshold_img = cv2.blur(facet_medians, ksize=(31, 31))

    bw_img = cv2.compare(img, threshold_img, cv2.CMP_GT)

    region_image, Cr = bw_area_detection(corners_subpix, bw_img)

    Fr, Rf = corner_region_association(corners_subpix, region_image, Cr)
    Rs = region_sides(Fr, Rf, Cr)

    offsets = None
    try:
        offsets = feature_offsets(corners_subpix, Fr, Rs, Cr)
    except ValueError or AssertionError as e:
        logger.debug(f"Unexpected topology: {e}")
        logger.debug(traceback.format_exc())

    if offsets is None:
        return np.array([]), np.array([])

    G, H = offsets
    indexed_corners = [i in G for i, _ in enumerate(corners_subpix)]

    objpoints = []
    imgpoints = []
    for idx in np.where(indexed_corners)[0]:
        imgpoints.append(np.array(corners_subpix[idx]) * frac)
        objpoints.append(G[idx] + (0, ))

    return np.array(objpoints), np.array(imgpoints)

# if __name__ == '__main__':
#     ### This is not working code, go to the notebooks/06-checkerboard.ipynb if you want extract calibration parameters
#     import os
#     from tqdm.auto import tqdm
#     ROOT = "/Users/gimli/cvr/data/zavity/trojan/02122024_sachovnice"
#     assert os.path.isdir(ROOT)
#
#     cap = cv2.VideoCapture(os.path.join(ROOT, "RED_sachovnice.MP4"))
#     valids = []
#     for frame_no in tqdm([900]):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
#         success, frame = cap.read()
#         if success:
#             try:
#                 objpoints, imgpoints = process_image(frame, 8)
#                 if objpoints.size >= 6 * 3:
#                     valids.append((frame_no, frame, objpoints, imgpoints))
#                 print(f"Frame {frame_no}: {len(objpoints)}")
#             except Exception as e:
#                 print(f"Frame {frame_no}: {e}")
#                 raise e