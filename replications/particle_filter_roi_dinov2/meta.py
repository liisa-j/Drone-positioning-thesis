import json
import os
import subprocess
import pyproj



# added this for exiftool path cofig
EXIFTOOL_PATH = os.environ.get(
    "EXIFTOOL_PATH",
    r"C:\Users\liisa\Desktop\exiftool-13.59_64\exiftool.exe",
)


# runs exiftool
def exiftool_paring(file_path):
    
    try:
        cmd = [
            EXIFTOOL_PATH,
            "-j",
            "-n",
            file_path,
        ]
        result = subprocess.check_output(cmd)
        data = json.loads(result)

        if data and isinstance(data, list):
            return data[0]

        return {}

    except Exception as e:
        print(f"ExifTool failed for {file_path}")
        print(f"{type(e).__name__}: {e}")
        raise

# extracts metadata
def read_exif_info(droonifail):    
    info = {
        "lat": None,
        "lon": None,
        "gps_alt": None,
        "rel_alt": None,
        "fov": None,
        "gimbal_yaw": None,
        "gimbal_pitch": None,
        "gimbal_roll": None,
        "flight_yaw": None,
        "flight_pitch": None,
        "flight_roll": None,
        "gps_heading": None,
        "lidar_range": None,
        "barometer_alt": None,
        "magnetic_heading": None,
    }

    meta = exiftool_paring(droonifail)

    if not meta:
        print("EXIF missing exiftool error:", droonifail)
        return info

    # GPS position and altitude
    lat = meta.get("GPSLatitude")
    lon = meta.get("GPSLongitude")
    gps_alt = meta.get("GPSAltitude")
    rel_alt = meta.get("RelativeAltitude")

    if lat is not None:
        info["lat"] = float(lat)
    if lon is not None:
        info["lon"] = float(lon)
    if gps_alt is not None:
        info["gps_alt"] = float(gps_alt)
    if rel_alt is not None:
        info["rel_alt"] = float(rel_alt)

    # Camera field of view
    fv = meta.get("Field Of View")
    if fv is not None:
        try:
            info["fov"] = float(fv)
        except (TypeError, ValueError):
            pass

    # Gimbal and flight orientation
    gy = meta.get("GimbalYawDegree")
    gp = meta.get("GimbalPitchDegree")
    gr = meta.get("GimbalRollDegree")

    fy = meta.get("FlightYawDegree")
    fp = meta.get("FlightPitchDegree")
    fr = meta.get("FlightRollDegree")

    if gy is not None:
        info["gimbal_yaw"] = float(gy)
    if gp is not None:
        info["gimbal_pitch"] = float(gp)
    if gr is not None:
        info["gimbal_roll"] = float(gr)

    if fy is not None:
        info["flight_yaw"] = float(fy)
    if fp is not None:
        info["flight_pitch"] = float(fp)
    if fr is not None:
        info["flight_roll"] = float(fr)

    # Other available measurements
    mg = meta.get("MagneticHeading")
    ba = meta.get("BarometerAltitude")
    gh = meta.get("GPSHeading")
    lr = meta.get("LidarRange")

    if mg is not None:
        info["magnetic_heading"] = float(mg)
    if ba is not None:
        info["barometer_alt"] = float(ba)
    if gh is not None:
        info["gps_heading"] = float(gh)
    if lr is not None:
        info["lidar_range"] = float(lr)

    return info

# converts coord sys from GPS to Estonian sys
def proj_3301(lat, lon):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3301",
        always_xy=True,
    )

    x, y = transformer.transform(lon, lat)
    return x, y


# img pixels to map coordinates (uses bounding box)
def pixel2lest(bbox, w, h, col, row):
    xmin, ymin, xmax, ymax = bbox

    if xmin > xmax:
        xmin, xmax = xmax, xmin

    if ymin > ymax:
        ymin, ymax = ymax, ymin

    x = xmin + (xmax - xmin) * (col / float(w))
    y = ymax - (ymax - ymin) * (row / float(h))

    return x, y


def get_projected_position_and_elevation(lat, lon, elevation_dataset):
    """Convert GPS coordinates to EPSG:3301 and retrieve terrain elevation."""
    x, y = proj_3301(lat, lon)

    for val in elevation_dataset.sample([(x, y)]):
        elevation = val[0]

    return x, y, elevation