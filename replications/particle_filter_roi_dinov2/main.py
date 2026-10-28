import os
import time
import numpy as np
import torch
import rasterio
from meta import read_exif_info, get_projected_position_and_elevation
from img_preprocess import (
    crop_and_rotate_roi,
    crop_black_edges,
    fix_yaw_if_nadir_roi,
    preprocess_drone_image,
    calculate_roi_edge_length,)
from feature_extr import (DinoExtractor, 
                          map_transform, 
                          drone_transform)
from matching import DINOEmbeddingMeasurement
from PF import (
    GPSStepDynamics,
    update_particles,
    estimate_position,
    randomize_roi_center,
    update_particle_population,)
from PIL import Image
from rasterio.transform import Affine
from pathlib import Path
from results import (
    get_unique_output_dir,
    CsvResultsWriter,
    save_particle_plot,
    save_trajectory_and_error_plots,
    calculate_offset,
)


def particle_filter_visual_demo(folder, 
                                m_name, 
                                extractor, 
                                ortho_img_path, 
                                elevation_path, 
                                flight='undefined', 
                                method='undefined', 
                                fov_hardcode = 55,
                                N_particles=500, 
                                scout_ratio=0.3, 
                                dynamic_noise_std=20.0, 
                                measurment_model_temp=0.07, 
                                create_figures=True):
    
    # look for drone imgs
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    if not files:
        print("Didn't find drone images:", folder)
        return

    #PF init
    N_scout_particles = int(N_particles * scout_ratio)
    particles = None  # will hold particles across iterations
    dynamics_model = GPSStepDynamics(noise_std=dynamic_noise_std)
    previous_drone_pos = None  # for GPS-based motion
    est_x, est_y = None, None

    #output dir
    session_output_dir = get_unique_output_dir(folder, flight, m_name, method)

    #result lists
    est_path = list()       # estimated positions [(x, y), ...]
    gps_path = list()       # true drone GPS positions [(x, y), ...]
    offsets = list()

    #ortophotos
    ortho_dataset = rasterio.open(ortho_img_path)
    ortho_img_rgb = np.moveaxis(ortho_dataset.read(), 0, -1)
    elevation_dataset = rasterio.open(elevation_path)

    # CSV setup
    csv_results = CsvResultsWriter(
        output_dir=session_output_dir,
        flight=flight,
        method=method,
        model_name=m_name,
        n_particles=N_particles,
        scout_ratio=scout_ratio,
        dynamic_noise_std=dynamic_noise_std,
        measurement_model_temp=measurment_model_temp,
    )

    #csv output path (print later)
    csv_path = csv_results.csv_path

    #main loop for pics
    for fn in files:
        path = os.path.join(folder, fn)
        print(f"\nProcessing: {path}")
        
        
        # === Metadata ===
        info = read_exif_info(path)
        lat, lon = info['lat'], info['lon']
        gps_alt = info['gps_alt']
        fov = info['fov'] if info['fov'] else fov_hardcode
        yaw, pitch, roll = info['gimbal_yaw'], info['gimbal_pitch'], info['gimbal_roll']

        #if lat or lon missing skip img
        if lat is None or lon is None:
            print("Missing lat/lon => skip.")
            continue

        #start time
        start_time = time.time()

        #get position and elev for ROI calculation
        xLe, yLe, elevation = get_projected_position_and_elevation(
            lat,
            lon,
            elevation_dataset,
            )
        ground_offset = elevation

        #start img preprocess (for rotate ROI)
        angle = fix_yaw_if_nadir_roi(yaw, pitch, roll, flight_yaw=info["flight_yaw"])

        #if first img, choose random ROI center, else ROI center is previous
        #estimated position
        if est_x is None or est_y is None:
            roi_center = randomize_roi_center(xLe, yLe, max_offset_m=50)
        else:
            roi_center = (est_x, est_y)


        #roi extraction and rotation here
        roi_rotated, roi_transform, _, roi_size = crop_and_rotate_roi(
            ortho_dataset, roi_center[0], roi_center[1], gps_alt, ground_offset, fov_deg=fov, yaw_deg=angle
        )

        #roi cropped and update geographic transform
        roi_rotated_cropped, roi_transform_cropped = crop_black_edges(roi_rotated, roi_transform)  
        roi_img = Image.fromarray(roi_rotated_cropped).resize((1024, 1024))
        scale_x = roi_rotated_cropped.shape[1] / 1024  # old_width / new_width
        scale_y = roi_rotated_cropped.shape[0] / 1024  # old_height / new_height
        # Update transform
        roi_resized_transform = roi_transform_cropped * Affine.scale(scale_x, scale_y)

        edge_length_m = calculate_roi_edge_length(roi_size)


        #particle update
        particles, previous_drone_pos = update_particles(
            particles=particles,
            N_particles=N_particles,
            roi_resized_transform=roi_resized_transform,
            xLe=xLe,
            yLe=yLe,
            previous_drone_pos=previous_drone_pos,
            dynamics_model=dynamics_model,
            )

        #drone img open and preprocess
        drone_img = preprocess_drone_image(path)

        
        # Extract dinov2 features from orto and drone photo (1, C, Hf, Wf)
        mapF = extractor.extract_features(roi_img, map_transform)
        droF = extractor.extract_features(drone_img, drone_transform)

        #convert particles to pytorch tensor
        particles_torch = torch.tensor(particles, dtype=torch.float32).unsqueeze(0)  # (1, M, 2)


        # --- Instantiate measurement model ---
        measurement_model = DINOEmbeddingMeasurement(
            mapF=mapF,
            transform=roi_resized_transform,
            drone_vec=droF,
            temperature=measurment_model_temp,
            oob_penalty=-6.0
        )

        # --- Compute particle weights ---
        logw = measurement_model.forward(particles_torch)  # (1, M)
        weights = torch.softmax(logw, dim=-1).squeeze(0).cpu().numpy()  # (M,)


        est_x, est_y = estimate_position(particles, weights)

        #time 
        inference_time_s = time.time() - start_time


        #plotting
        if create_figures:
            save_particle_plot(
                        roi_img=roi_img,
                        particles=particles,
                        roi_transform=roi_resized_transform,
                        est_x=est_x,
                        est_y=est_y,
                        drone_x=previous_drone_pos[0],
                        drone_y=previous_drone_pos[1],
                        drone_img=drone_img,
                        image_name=fn,
                        output_dir=session_output_dir,
                    )


        #update particles
        particles = update_particle_population(
            particles=particles,
            weights=weights,
            N_particles=N_particles,
            N_scout_particles=N_scout_particles,
            roi_resized_transform=roi_resized_transform,
            )


        #store estimated position and ref GPS
        est_path.append((est_x, est_y))
        gps_path.append(previous_drone_pos)
        
    
        #calculate and store offset (error)
        offset_m = calculate_offset(
            est_x,
            est_y,
            previous_drone_pos[0],
            previous_drone_pos[1],
            )
        offsets.append(offset_m)
        
        # Write CSV row
        csv_results.write_row(fn, offset_m, edge_length_m, inference_time_s)
 
    
#out of img processing loop...

    #close writer and say where the results are
    csv_results.close()
    print(f"Results saved to {csv_path}")

    #final plots
    if create_figures:
        save_trajectory_and_error_plots(
            est_path=est_path,
            gps_path=gps_path,
            offsets=offsets,
            ortho_dataset=ortho_dataset,
            ortho_img_rgb=ortho_img_rgb,
            output_dir=session_output_dir,
            flight=flight,
            method=method,
            model_name=m_name,
        )
        
    return session_output_dir, offsets

#file location
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def calculate_locations(
    flight,
    ortho_img_filepath,
    elevation_path,
    model_name,
    extractor,
    method,
    fov_hardcode=55,
    ):
    folder = PROJECT_ROOT / "droonifotod"

    particle_filter_visual_demo(
        str(folder),
        model_name,
        extractor,
        str(ortho_img_filepath),
        str(elevation_path),
        flight=flight,
        method=method,
        fov_hardcode=fov_hardcode,
        N_particles=500,
        scout_ratio=0,
        dynamic_noise_std=10,
        measurment_model_temp=0.07,
    )


if __name__ == "__main__":
    model = "dinov2-base"
    extractor = DinoExtractor(model)

    calculate_locations(
        flight="droonifotod",
        ortho_img_filepath=PROJECT_ROOT / "ortofoto" / "54752.tif",
        elevation_path=PROJECT_ROOT / "ortofoto" / "54752_dtm_1m.tif",
        model_name=model,
        extractor=extractor,
        method="NEWWW----pf-test",
        fov_hardcode=37,
    )