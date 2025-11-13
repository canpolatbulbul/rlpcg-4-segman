import csv
import robotic as ry
import sys
sys.path.append('..')
from MOSeGMan.MOSeGMan import MOSeGMan
import os
import random


random.seed(44)
verbose = 0

ry.params_clear()
ry.params_add({"Render/lights":[-5.,0.,5., -5.,0.,5.]})
base_path = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":
    case_id = 79
    folder = f"ry_config/case_run_id_{case_id}/"
    # Get the names of all folders in the directory that start with "pcg-"
    pcg_folders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name)) and name.startswith("pcg-")]

    # Create a .csv file named "tracks.csv" with columns: track_id, time, x, y, vx, vy
    with open(f"data/tracks_id_{case_id}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["track_id", "time", "x", "y", "vx", "vy"])

        for i, folder_name in enumerate(pcg_folders):
            C = ry.Config()
            C_hm = ry.Config()
            C.addFile(os.path.join(folder, f"{folder_name}/{folder_name}.g"))
            C_hm.addFile(os.path.join(folder, f"{folder_name}/{folder_name}-aux.g"))
            segman = MOSeGMan(C, C_hm, verbose=verbose)
            if segman.run():
                Ct = ry.Config()
                Ct.addConfigurationCopy(segman.C)
                FS = segman.segman.FS
                px, py = None, None
                dt = 0.01
                for j, fs in enumerate(FS):
                    Ct.setFrameState(fs)
                    track_id = i
                    time = j*dt
                    pos = Ct.frame("ego").getPosition()
                    x, y = pos[0], pos[1]
                    if px is None and py is None:
                        vx, vy = 0.0, 0.0
                    else:
                        vx = (x - px) / dt
                        vy = (y - py) / dt
                    px, py = x, y
                    writer.writerow([track_id, time, x, y, vx, vy])



            






