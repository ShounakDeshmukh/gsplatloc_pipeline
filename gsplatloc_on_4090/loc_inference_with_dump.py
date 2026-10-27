
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# loc_inference_with_dump.py: GSplatLoc's loc_inference.py with these additions (marked "ADDED"):
#   * query images are moved to the GPU one at a time, so a model trained with --data_device cpu works
#   * a failed PnP no longer crashes the run; it is counted as a failure
#   * per-query results go to <model>/error_logs/per_query_results.json
#   * every --save_every-th query is saved as  query | render at the estimated pose  in <model>/loc_examples/
# The localization method itself is unchanged. Copy this file into the GSplatLoc repo root to use it.
import cv2
import json
import os
import numpy as np
import time
import torch
import torch.optim as optim

from scene import Scene
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, fov2focal

from encoders.XFeat.modules.xfeat import XFeat

from warping.warping_loss import *
from warping.warp_utils import *
from utils.loc_utils import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def localize_set(model_path, name, views, gaussians, pipeline, background, args):

        # Keep track of rotation and translation errors for calculation of the median error.
        rErrs = []
        tErrs = []

        prior_rErr = []
        prior_tErr = []

        xfeat = XFeat()
        records, n_failed = [], 0  # ADDED
        os.makedirs(os.path.join(model_path, "loc_examples"), exist_ok=True)  # ADDED

        gaussian_pcd = gaussians.get_xyz
        gaussian_feat = gaussians.get_semantic_feature.squeeze(1)
            
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            gt_im = view.original_image[0:3, :, :].cuda()  # ADDED .cuda()
            t_query = time.time()  # ADDED
            
            # Extract sparse features
            gt_keypoints, _, gt_feature = xfeat.detectAndCompute(gt_im[None], 
                                                                 top_k=args.top_k)[0].values()

            # Define intrinsic matrix
            K = np.eye(3)
            focal_length = fov2focal(view.FoVx, view.image_width)
            K[0, 0] = K[1, 1] = focal_length
            K[0, 2] = view.image_width / 2
            K[1, 2] = view.image_height / 2

            start = time.time()

            # Find initial pose prior via 2D-3D matching
            with torch.no_grad():
                matched_3d, matched_2d = find_2d3d_correspondences(
                    gt_keypoints,
                    gt_feature,
                    gaussian_pcd,
                    gaussian_feat
                )

            gt_R = view.R
            gt_t = view.T

            print(f"Match speed: {time.time() - start}")
            try:  # ADDED: do not crash on a failed PnP
                pnp_ok, R, t, inliers = cv2.solvePnPRansac(matched_3d, matched_2d,
                                                  K,
                                                  distCoeffs=None,
                                                  flags=cv2.SOLVEPNP_ITERATIVE,
                                                  iterationsCount=args.ransac_iters
                                                  )
            except cv2.error:
                pnp_ok, R, t, inliers = False, None, None, None
            if not pnp_ok or R is None:
                n_failed += 1
                records.append(dict(name=view.image_name, ok=False, matches=int(len(matched_2d))))
                print(f"PnP failed for {view.image_name}")
                continue
            n_inliers = 0 if inliers is None else int(len(inliers))

            R, _ = cv2.Rodrigues(R)

            # Calculate the rotation and translation errors using existing function
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R.T, t)

            # Print the errors
            print(f"Rotation Error: {rotError} deg")
            print(f"Translation Error: {transError} cm")

            prior_rErr.append(rotError)
            prior_tErr.append(transError)
            prior_rot, prior_trans = float(rotError), float(transError)  # ADDED

            w2c = torch.eye(4, 4, device='cuda')
            w2c[:3, :3] = torch.from_numpy(R).float()
            w2c[:3, 3] = torch.from_numpy(t[:, 0]).float()
            
            # Update the view's pose
            view.update_RT(R.T, t[:,0])
            
            # Render from the current estimated pose
            with torch.no_grad():
                render_pkg = render(view, gaussians, pipeline, background)
            
            render_im = render_pkg["render"]
            depth = render_pkg["depth"]

            quat_opt = rotmat2qvec_tensor(w2c[:3, :3].clone()).view([4]).to(w2c.device)
            t_opt = w2c[:3, 3].clone()

            optimizer = optim.Adam([quat_opt.requires_grad_(True), 
                                    t_opt.requires_grad_(True)], lr=args.warp_lr)


            for i in range(args.warp_iters):                    
                    
                # Compute warp loss for optimizing w2c_opt
                optimizer.zero_grad()
       
                loss = compute_warping_loss(vr=render_im,
                                            qr=gt_im,
                                            quat_opt=quat_opt,
                                            t_opt=t_opt,
                                            pose=w2c,
                                            K=torch.from_numpy(K).float().to('cuda'),
                                            depth=depth) 
        
                loss.backward()
                optimizer.step()
                
                if i % (args.warp_iters // 5) == 0:
                    print(f"Iteration {i}, Loss: {loss.item():.4f}")
                    
                    # After optimization, update the view's pose
                    R_est = qvec2rotmat_tensor(quat_opt).detach().cpu().numpy()
                    t_est = t_opt.detach().cpu().numpy()

                    # Compute final errors
                    rotError, transError = calculate_pose_errors(gt_R, gt_t, R_est.T, t_est.reshape(3,1))

                    print(f"Iteration {i} Rotation Error: {rotError:.2f} deg, Translation Error: {transError:.2f} cm")
                
            # After optimization, update the view's pose
            R_est = qvec2rotmat_tensor(quat_opt).detach().cpu().numpy()
            t_est = t_opt.detach().cpu().numpy()
            
            # Compute final errors
            rotError, transError = calculate_pose_errors(gt_R, gt_t, R_est.T, t_est.reshape(3,1))

            print(f"Final Rotation Error: {rotError:.2f} deg, Translation Error: {transError:.2f} cm")
            
            rErrs.append(rotError)
            tErrs.append(transError)

            # ADDED: record the result and, every few queries, save query | render at the estimated pose
            seconds = time.time() - t_query
            records.append(dict(name=view.image_name, ok=True, matches=int(len(matched_2d)), pnp_inliers=n_inliers,
                                prior_rot_err_deg=prior_rot, prior_trans_err_cm=prior_trans,
                                rot_err_deg=float(rotError), trans_err_cm=float(transError), seconds=seconds,
                                R_world_to_cam=R_est.tolist(), t_world_to_cam=t_est.reshape(3).tolist()))
            if args.save_every > 0 and len(records) % args.save_every == 0:
                try:
                    view.update_RT(R_est.T, t_est.reshape(3))
                    with torch.no_grad():
                        out_im = render(view, gaussians, pipeline, background)["render"]
                    to8 = lambda x: (x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
                    pair = np.hstack([to8(gt_im), to8(out_im)]).copy()
                    cv2.putText(pair, f"query {view.image_name} | render at estimated pose: {transError:.1f} cm, {rotError:.2f} deg",
                                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imwrite(os.path.join(model_path, "loc_examples", f"{view.image_name}.jpg"), pair)
                except Exception as e:
                    print(f"could not save example image: {e}")

            print(f"Processed: {view.uid}")            
        
        log_errors(model_path, name, prior_rErr, prior_tErr, f"prior")
        log_errors(model_path, name, rErrs, tErrs, "warp")
        # ADDED
        with open(os.path.join(model_path, "error_logs", "per_query_results.json"), "w") as f:
            json.dump(records, f, indent=1)
        good = [r for r in records if r["ok"]]
        print(f"queries {len(records)}, PnP failures {n_failed}, "
              f"median time per query {np.median([r['seconds'] for r in good]) if good else float('nan'):.2f} s")
        

def launch_inference(dataset : ModelParams, pipeline : PipelineParams, args): 
         
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    localize_set(dataset.model_path, "test", scene.getTestCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--top_k", default=1_000, type=int)
    parser.add_argument("--ransac_iters", default=20000, type=int)
    parser.add_argument("--warp_lr", default=0.0005, type=float)
    parser.add_argument("--warp_iters", default=251, type=int)
    parser.add_argument("--save_every", default=5, type=int)  # ADDED: 0 disables the example images
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch_inference(model.extract(args), pipeline.extract(args), args)