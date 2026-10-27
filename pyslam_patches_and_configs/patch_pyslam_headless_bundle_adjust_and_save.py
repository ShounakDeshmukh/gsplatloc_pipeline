#!/usr/bin/env python3
"""Make `main_slam.py --headless` do what the Map Viewer buttons do at the end of a run: wait for
local mapping and loop closing to settle, Bundle Adjust, Save (map.json). Stock headless exits
without either. Run inside the container after each start; idempotent.
usage: python ~/qualcomm_evk_loc_demo/pyslam_patches_and_configs/patch_pyslam_headless_bundle_adjust_and_save.py [~/pyslam]"""
import os, sys

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/pyslam")
path = os.path.join(root, "main_slam.py")
src = open(path).read()
if "HEADLESS_BA_AND_SAVE" in src:
    sys.exit("already patched")
anchor = "    # exit from the main loop\n"
if src.count(anchor) != 1:
    sys.exit(f"expected 1 anchor line, found {src.count(anchor)}: not patching")
block = '''
    # HEADLESS_BA_AND_SAVE (added by patch_pyslam_headless_bundle_adjust_and_save.py)
    if args.headless:
        try:
            t_wait = time.time()
            n_quiet = 0
            while n_quiet < 10 and time.time() - t_wait < 600:  # 2 s of quiet, give up after 10 min
                busy = slam.local_mapping.queue_size() > 0 or not slam.local_mapping.is_idle()
                lc = getattr(slam, "loop_closing", None)
                gba = getattr(lc, "GBA", None) if lc is not None else None
                if gba is not None and (gba.is_running() or gba.is_correcting()):
                    busy = True
                n_quiet = 0 if busy else n_quiet + 1
                time.sleep(0.2)
            Printer.green(f"HEADLESS: backend quiet after {time.time() - t_wait:.1f} s, frames {num_frames}")
            slam.bundle_adjust()
            Printer.green("HEADLESS: bundle adjust returned")
        except Exception as e:
            Printer.red(f"HEADLESS: bundle adjust FAILED, saving the map anyway: {e}")
            print(traceback.format_exc())
        try:
            slam.save_system_state(config.system_state_folder_path)
            dataset.save_info(config.system_state_folder_path)
            groundtruth.save(config.system_state_folder_path)
            Printer.green(f"HEADLESS: MAP SAVED to {config.system_state_folder_path}")
        except Exception as e:
            Printer.red(f"HEADLESS: MAP SAVE FAILED: {e}")
            print(traceback.format_exc())
'''
open(path, "w").write(src.replace(anchor, anchor + block))
print(f"patched {path}")
