import math
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import Affine
from PIL import Image, ImageOps
from shapely.geometry import box
from shapely import affinity



#calculate the correct angle by which to rotate the ROI
def fix_yaw_if_nadir_roi(yaw, pitch, roll, flight_yaw=None):
    fix = 0
    if yaw is None and flight_yaw is None:
        return None

    if pitch is not None and abs(pitch + 90) < 10:
        angle = -yaw  if yaw is not None else 0.0
        if flight_yaw is not None:
            yaw_diff = abs((yaw - flight_yaw + 360) % 360)
            if 160 <= yaw_diff <= 200:  
                print(f"Yaw-flip detected (gimbal: {yaw:.1f}, flight: {flight_yaw:.1f}) => flip +180°")
                angle = (angle + 180) % 360 + fix
        print(f"Rotating drone image {angle:.1f} degrees (nadir, gimbal yaw)")
        return angle

    elif flight_yaw is not None:
        angle = - flight_yaw + fix
        #angle = flight_yaw
        print(f"Rotating drone image {angle:.1f} degrees (flight yaw), pitch={pitch}")
        return angle

    elif yaw is not None:
        angle = -yaw
        #angle = yaw 
        print(f"Rotating drone image {angle:.1f} degrees (fallback gimbal yaw), pitch={pitch}")
        return angle

    return yaw


#crop an area of the ortophoto (around est drone pos) 
# and rotate it (to match cam yaw)
def crop_and_rotate_roi(ortho_image, x_center, y_center, gps_alt, ground_offset,
                        fov_deg=55.0, yaw_deg=0.0):
    """
    Crop a rotated ROI from a GeoTIFF orthophoto and rotate it to align with drone yaw.
    
    Returns:
        - rotated_image (HWC numpy array)
        - rotated_transform (affine)
        - out_crs (CRS)
        - roi_corners (list of 4 (x,y) tuples in map coords, EPSG:3301)
    """
    # --- 1. Compute ROI size ---
    alt_p = gps_alt - ground_offset if gps_alt else 200.0
    print(f"altp: {alt_p}")
    alt_p = max(alt_p, 30.0)  # minimum altitude
    print(f"altp: {alt_p}")

    print(f"fov: {fov_deg}")
    width = 2.0 * alt_p * math.tan(math.radians(fov_deg / 2.0)) * 4
    print(f"width: {width}")
    height = width  # square ROI

    # --- 2. Create rotated ROI polygon ---
    roi_box = box(x_center - width / 2, y_center - height / 2,
                  x_center + width / 2, y_center + height / 2)
    roi_rot = affinity.rotate(roi_box, yaw_deg, origin=(x_center, y_center), use_radians=False)
    roi_corners = list(roi_rot.exterior.coords)[:-1]  # drop duplicate last point
    roi_size = abs(roi_corners[0][0]-roi_corners[2][0]) * abs(roi_corners[0][1]-roi_corners[2][1])

    # --- 3. Crop raster ---
    out_image, out_transform = rasterio.mask.mask(ortho_image, [roi_rot], crop=True)
    out_image = np.transpose(out_image, (1, 2, 0))  # bands-first → HWC
    out_meta = ortho_image.meta.copy()
    out_crs = out_meta["crs"]

    # --- 4. Rotate cropped image ---
    h, w = out_image.shape[:2]
    theta = math.radians(-yaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Rotate image using PIL
    from PIL import Image
    img_pil = Image.fromarray(out_image.astype(np.uint8))
    rotated_pil = img_pil.rotate(-yaw_deg, resample=Image.Resampling.BILINEAR, expand=False) # we need to turn in the opposite direction of the roi rotation
    rotated_image = np.array(rotated_pil)


    #rotate the Affine
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    r11 = cos_t
    r12 = -sin_t
    r21 = sin_t
    r22 = cos_t

    tx = cx - (r11 * cx + r12 * cy)
    ty = cy - (r21 * cx + r22 * cy)

    T = Affine(r11, r12, tx, r21, r22, ty)

    rotated_transform = out_transform * T

    return rotated_image, rotated_transform, out_crs, roi_size


# remove possible black edges after rotation - possible problem over dark areas?
def crop_black_edges(img_array, transform, tol=5):
    """s
    Crop black edges from an image (numpy array HWC, 0-255).
    tol: pixel values below tol are considered black.
    Returns the cropped array.
    """
    if img_array.ndim == 3:
        mask = (img_array > tol).any(axis=2)
    else:
        mask = img_array > tol

    if not mask.any():
        # Everything is black
        return img_array, transform

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slice end is exclusive
    cropped = img_array[y0:y1, x0:x1]
    new_transform = transform * Affine.translation(x0, y0)

    return cropped, new_transform


def preprocess_drone_image(image_path):
    """Load and preprocess a drone image for feature extraction."""
    drone_img = Image.open(image_path).convert("RGB")
    drone_img = ImageOps.fit(
        drone_img,
        (256, 256),
        Image.Resampling.BILINEAR,
    )
    return drone_img


def calculate_roi_edge_length(roi_size):
    """Calculate the ROI edge length in meters."""
    return math.sqrt(abs(roi_size))