from functools import partial
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import box, shape
from shapely.affinity import translate
import numpy as np
from rasterio import features, Affine
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer
from numba import jit
from skimage import segmentation, transform, exposure, measure
from scipy import ndimage, LowLevelCallable
import tqdm
import requests
import json
import os

PINK = np.array([255, 15, 255]) / 255.
YELLOW = np.array([255, 255, 15]) / 255.


@jit
def tss(a):

    # total sum of square difference
    total_sum_of_squares = np.sum((a - np.mean(a)) ** 2)

    return total_sum_of_squares


@cfunc(intc(CPointer(float64), intp,
            CPointer(float64), voidptr))
def nbtss(values_ptr, len_values, result, data):

    # total sum of square difference (C-implementation for speedup)
    values = carray(values_ptr, (len_values,), dtype=float64)
    sum = 0.0
    for v in values:
        sum += v
    mean = sum / float64(len_values)
    result[0] = 0
    for v in values:
        result[0] += (v - mean) ** 2

    return 1


def bbox2d(img):

    # identify the bounding box of elements in the array that are >0
    img = (img > 0)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
    cmin, cmax = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))

    return rmin, rmax, cmin, cmax


def collar_mask(img):

    # return a collar mask for the input image. in a moving window search where the non-zero
    # elements of the image are used as the window, the collar mask will show all parts along
    # the edge of the image where the window would cross over the image edge
    rmin, rmax, cmin, cmax = bbox2d(img)
    height = rmax - rmin
    width = cmax - cmin
    half_height = int(np.ceil(height / 2.))
    half_width = int(np.ceil(width / 2.))
    collar_mask = np.zeros(img.shape, dtype='int')
    collar_mask[:half_height, :] = 1
    collar_mask[-half_height:, :] = 1
    collar_mask[:, :half_width] = 1
    collar_mask[:, -half_width:] = 1

    return collar_mask


def geom_to_array(geom, img, geom_val=1, fill_value=0, all_touched=True):

    # convert geometry object to an array, using the affine from the input image
    geom_array = features.rasterize([(geom, geom_val)],
                                    out_shape=(img.shape[1], img.shape[2]),
                                    transform=img.affine,
                                    fill=fill_value,
                                    all_touched=all_touched,
                                    dtype=np.uint8)

    return geom_array


def labels_to_polygons(labels_array, image_affine, ignore_label=0):

    # create polygon generator object
    polygon_generator = features.shapes(labels_array.astype('uint8'),
                                        mask=labels_array != ignore_label,
                                        transform=image_affine)
    # Extract out the individual polygons, fixing any invald geometries using buffer(0)
    polygons = [{'geometry': shape(g).buffer(0), 'properties': {'id': v}} for g, v in polygon_generator]

    return polygons


def plot_array(array, subplot_ijk, title="", font_size=18, cmap=None):

    sp = plt.subplot(*subplot_ijk)
    sp.set_title(title, fontsize=font_size)
    plt.axis('off')
    plt.imshow(array, cmap=cmap)


def translate_geom(row, xoff, yoff):

    return translate(row['geometry'], xoff=row[xoff], yoff=row[yoff])


def calc_translation(geom, image, search_buffer=0.0001, le90=0.00003, downscale=3, weights=[1, 1, 1]):

    # buffer the image out by the search buffer and define the chip
    bounds = geom.buffer(search_buffer).bounds
    chip = image.aoi(bbox=bounds)

    # extract out the rgb bands
    rgb = chip.base_layer_match(blm=True)
    # convert the geom to an array, aligned with the rgb
    bldg_array = geom_to_array(geom, chip)
    # also define the area within which we will allow the geometry's centroid to be shifted, defined
    # as the le90 buffer around the original geom centroid. make this an array too
    shift_area_array = geom_to_array(geom.centroid.buffer(le90), chip)

    # downscale the three arrays from above (for faster performance at little cost to quality of results)
    rgb = transform.pyramid_reduce(rgb, downscale=downscale, multichannel=True)
    bldg_array = transform.pyramid_reduce(bldg_array, downscale=downscale, multichannel=False) > 0
    shift_area_array = transform.pyramid_reduce(shift_area_array, downscale=downscale, multichannel=False) == 0

    # define the allow able areas where the image can NOT be moved, defined as the boolean OR of the
    # collar mask and the shift area array
    mask = ((collar_mask(bldg_array)) | (shift_area_array)) * -9999

    # step 1 -- using a moving window of the building array, determine areas where the ndvi is in the lower
    # 50% of the chip
    ndvi = chip.ndvi()
    ndvi[np.isnan(ndvi)] = 0
    ndvi_rescale = exposure.rescale_intensity(ndvi, out_range=(-1, 1))
    ndvi_downsamp = transform.pyramid_reduce(ndvi_rescale, downscale=downscale, multichannel=False)
    ndvi_results = ndimage.generic_filter(ndvi_downsamp, np.mean, footprint=bldg_array, mode='constant', cval=-9999)
    ndvi_results_masked = np.ma.masked_where(mask == -9999, ndvi_results)
    ndvi_min = np.ma.less_equal(ndvi_results_masked, np.percentile(ndvi_results_masked.compressed(), 50))

    # step 2 -- using a moving window of the building array, determine areas where the BAI is in the upper
    # 25% of the chip
    bai = (chip[1, :, :] - chip[6, :, :]) / (chip[1, :, :] + chip[6, :, :])
    bai[np.isnan(bai)] = 0
    bai_rescale = exposure.rescale_intensity(bai, out_range=(-1, 1))
    bai_downsamp = transform.pyramid_reduce(bai_rescale, downscale=downscale, multichannel=False)
    bai_results = ndimage.generic_filter(bai_downsamp, np.mean, footprint=bldg_array,
                                         mode='constant', cval=-9999)
    bai_results_masked = np.ma.masked_where(mask == -9999, bai_results)
    bai_max = np.ma.greater_equal(bai_results_masked, np.percentile(bai_results_masked.compressed(), 75))

    # step 3 -- using a moving window of the building array, determine areas where the total sum of squares
    # difference of the RGB channels is in the lower 50% of the chip
    tss_results = np.zeros(rgb.shape, dtype='float')
    for i in range(0, rgb.shape[2]):
        tss_results[:, :, i] = ndimage.generic_filter(rgb[:, :, i], LowLevelCallable(nbtss.ctypes),
                                                      footprint=bldg_array,
                                                      mode='constant', cval=-9999)
    mask = np.ones(rgb.shape) * mask[:, :, None]
    tss_results_masked = np.ma.masked_where(mask == -9999, tss_results)
    tss_sum = np.ma.sum(tss_results, axis=2)
    tss_sum.mask = mask[:, :, 0]
    tss_min = np.ma.less_equal(tss_sum, np.percentile(tss_sum.compressed(), 50))

    # add up the three components identified above, weighted by the input weights
    potential_centroids = (ndvi_min.astype('int') * weights[0] + tss_min.astype('int') * weights[1]
                           + bai_max.astype('int') * weights[2])
    # identify the areas where the score is highest for these three components
    top_centroids = potential_centroids == np.ma.max(potential_centroids)

    # extract a centroid for each of the candidate best regions
    downsampled_affine = Affine(chip.affine.a * downscale, chip.affine.b, chip.affine.c, chip.affine.d,
                                chip.affine.e * downscale, chip.affine.f)
    new_locs = labels_to_polygons(measure.label(top_centroids), downsampled_affine, ignore_label=0)
    new_centroids, new_areas, new_dists = zip(
        *[(new_loc['geometry'].centroid, new_loc['geometry'].area, new_loc['geometry'].distance(geom.centroid))
          for new_loc in new_locs])

    # pick the best region by shorted distance from the original centroid
    new_centroid = new_centroids[np.argmin(new_dists)]
    xoff, yoff = np.array(new_centroid.coords[0]) - np.array(geom.centroid.coords[0])

    return xoff, yoff


def align_to_image(geoms_df, image, le90=0.00003, search_buffer=0.0001, downscale=3, progress=True):
    # buffer in from the image bounds to make sure we don't move any geoms too close to the edge
    image_geom_bfr = shapely.geometry.box(*image.bounds).buffer(-1 * search_buffer)

    geoms_df['xoff'] = np.float64(0)
    geoms_df['yoff'] = np.float64(0)
    if progress is True:
        progress_func = partial(tqdm.tqdm_notebook, total=len(geoms_df))
    else:
        progress_func = lambda x: x

    for i, row in progress_func(geoms_df.iterrows()):
        geom = row['geometry']
        if geom.within(image_geom_bfr):
            xoff, yoff = calc_translation(geom, image, le90=le90, search_buffer=search_buffer,
                                          downscale=downscale, weights=[1, 1, 1])
            geoms_df.loc[i, 'xoff'] = xoff
            geoms_df.loc[i, 'yoff'] = yoff

    results_df = geoms_df.copy()
    results_df['shifted_geom'] = results_df.apply(func=translate_geom, axis=1, args=('xoff', 'yoff'))

    return results_df


def from_geojson(source):
    if source.startswith('http'):
        response = requests.get(source)
        geojson = json.loads(response.content)
    else:
        if os.path.exists(source):
            with open(source, 'r') as f:
                geojson = json.loads(f.read())
        else:
            raise ValueError("File does not exist: {}".format(source))

    geometries = []
    feats = []
    for f in geojson['features']:
        geom = geometry.shape(f['geometry'])
        feats.append({'geometry': geom, 'properties': {}})
        geometries.append(geom)

    return geometries, feats