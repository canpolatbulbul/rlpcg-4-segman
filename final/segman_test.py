import csv
import robotic as ry
import sys
import time

sys.path.append("..")
from MOSeGMan.MOSeGMan import MOSeGMan
import os
import random


random.seed(44)
# verbose is now set via command line argument

ry.params_clear()
ry.params_add({"Render/lights": [-5.0, 0.0, 5.0, -5.0, 0.0, 5.0]})
base_path = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Test SeGMaN on generated .g files")
    ap.add_argument("--case_id", type=int, default=1, help="Case ID to test")
    ap.add_argument(
        "--view",
        action="store_true",
        help="View config before solving (pauses for each config)",
    )
    ap.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level (0=silent, 1=print planning info, 2+=more debug)",
    )
    ap.add_argument(
        "--replay",
        action="store_true",
        help="Replay solution animation after finding a solution",
    )
    ap.add_argument(
        "--replay_speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (higher = faster, default: 1.0)",
    )
    args = ap.parse_args()

    case_id = args.case_id
    verbose = args.verbose
    
    # Handle special test case
    if str(case_id) == "test_movable":
        folder = "ry_config/test_movable/"
        # Create test config if it doesn't exist
        if not os.path.exists(f"{folder}test_movable.g"):
            print("Creating test movable config...")
            from test_movable_config import create_test_movable_config
            create_test_movable_config(size=17, output_dir=folder)
        pcg_folders = []  # Will handle as single file case
    else:
        folder = f"ry_config/case_run_id_{case_id}/"
        # Get the names of all folders in the directory that start with "pcg-"
        pcg_folders = [
            name
            for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name)) and name.startswith("pcg-")
        ]

    # Create a .csv file named "tracks.csv" with columns: track_id, time, x, y, vx, vy
    with open(f"data/tracks_id_{case_id}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["track_id", "time", "x", "y", "vx", "vy"])

        # Handle test_movable case (single file, not a folder)
        if str(case_id) == "test_movable":
            test_files = ["test_movable"]
        else:
            test_files = pcg_folders
        
        for i, folder_name in enumerate(test_files):
            print(f"\n=== Testing {folder_name} ===")
            C = ry.Config()
            C_hm = ry.Config()
            
            if str(case_id) == "test_movable":
                # Single file case
                C.addFile(os.path.join(folder, f"{folder_name}.g"))
                C_hm.addFile(os.path.join(folder, f"{folder_name}-aux.g"))
            else:
                # Folder case
                C.addFile(os.path.join(folder, f"{folder_name}/{folder_name}.g"))
                C_hm.addFile(os.path.join(folder, f"{folder_name}/{folder_name}-aux.g"))

            # View config before solving if requested
            if args.view:
                print(
                    f"Viewing config {folder_name}... (close viewer window to continue)"
                )
                C.view(pause=True, message=f"PCG Config: {folder_name}")
            C.view(True)
            segman = MOSeGMan(C, C_hm, verbose=verbose)
            print(f"Running MOSeGMan on {folder_name}...")
            success = segman.run()
            if success:
                print(f"✓ Solution found for {folder_name}")
                Ct = ry.Config()
                Ct.addConfigurationCopy(segman.C)
                # Use segman.segman.FS which should contain the complete solution
                # (self.FS in SeGManv2 accumulates across all solve() calls and isn't reset)
                FS = segman.segman.FS
                print(f"Using segman.segman.FS: {len(FS)} states")
                print(f"Number of objects to solve: {len(segman.obj_g_list)}")
                
                px, py = None, None
                dt = 0.01
                
                # Replay solution if requested
                if args.replay:
                    print(f"Replaying solution for {folder_name} ({len(FS)} steps)...")
                    # Open viewer once at the start
                    Ct.view(False, message=f"Solution Replay: {folder_name}")
                
                for j, fs in enumerate(FS):
                    Ct.setFrameState(fs)
                    
                    # Update visualization during replay (no pause, auto-updates)
                    if args.replay:
                        # Update viewer without pausing (False = no pause)
                        Ct.view(False, message=f"Step {j+1}/{len(FS)}")
                        # Small delay to make animation visible (scaled by replay_speed)
                        time.sleep(dt / args.replay_speed)
                    
                    # Write to CSV
                    track_id = i
                    time_val = j * dt
                    pos = Ct.frame("ego").getPosition()
                    x, y = pos[0], pos[1]
                    if px is None and py is None:
                        vx, vy = 0.0, 0.0
                    else:
                        vx = (x - px) / dt
                        vy = (y - py) / dt
                    px, py = x, y
                    writer.writerow([track_id, time_val, x, y, vx, vy])
                
                if args.replay:
                    print(f"Replay complete for {folder_name}")
            else:
                print(f"✗ No solution found for {folder_name}")
