#!/usr/bin/env python3
"""
test_movable_config.py - Generate a test config with objects on right, goals on left,
and movables blocking the middle path.

This creates a clear test case where movables MUST be moved to solve the puzzle.
"""

import robotic as ry
import numpy as np

def create_test_movable_config(size=17, output_dir="ry_config/test_movable"):
    """
    Create a test config:
    - Objects on right half
    - Goals on left half  
    - Long wall in middle
    - Middle section of wall is movables (must be moved to create path)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create grid
    grid = np.zeros((size, size), dtype=int)
    
    # Constants
    EMPTY, WALL, ROBOT, OBJECT, GOAL, MOVABLE = 0, 1, 2, 3, 4, 5
    
    # Place robot on right side
    robot_y, robot_x = size // 2, size - 2
    grid[robot_y, robot_x] = ROBOT
    
    # Place objects on right half
    obj1_y, obj1_x = size // 4, size - 3
    obj2_y, obj2_x = 3 * size // 4, size - 3
    grid[obj1_y, obj1_x] = OBJECT
    grid[obj2_y, obj2_x] = OBJECT
    
    # Place goals on left half
    goal1_y, goal1_x = size // 4, 2
    goal2_y, goal2_x = 3 * size // 4, 2
    grid[goal1_y, goal1_x] = GOAL
    grid[goal2_y, goal2_x] = GOAL
    
    # Create long wall in middle (vertical wall at x = size//2)
    wall_x = size // 2
    wall_start = size // 4
    wall_end = 3 * size // 4
    
    # Top part: solid wall
    for y in range(0, size // 2 - 2):
        grid[y, wall_x] = WALL
    
    # Middle part: MOVABLES (this is the key - must be moved!)
    movable_start = size // 2 - 2
    movable_end = size // 2 + 2
    for y in range(movable_start, movable_end):
        grid[y, wall_x] = MOVABLE
    
    # Bottom part: solid wall
    for y in range(movable_end, size):
        grid[y, wall_x] = WALL
    
    # Convert to rai config
    env_size = 4.0
    wall_thickness = 0.1
    ob_s = (env_size - wall_thickness) / float(size)
    sx = -(env_size / 2) + ob_s / 2 + 0.05
    sy = (env_size / 2) - ob_s / 2 - 0.05
    
    C = ry.Config()
    C.addFile("ry_config/base.g")
    C_aux = ry.Config()
    C_aux.addFile("ry_config/base-aux.g")
    
    # Track indices
    object_count = 0
    goal_count = 0
    movable_count = 0
    
    for r in range(size):
        for c in range(size):
            tile = int(grid[r, c])
            x_pos = sx + ob_s * c
            y_pos = sy - ob_s * r
            
            if tile == WALL:
                f = C.addFrame(f"block_{r}_{c}", "world", 
                             f"shape:ssBox, size:[{ob_s}, {ob_s}, 0.2, 0.01], color:[0.6953, 0.515625, 0.453125], contact:1")
                f.setRelativePosition([x_pos, y_pos, 0.1])
                f_aux = C_aux.addFrame(f"block_{r}_{c}", "world", 
                                     f"shape:ssBox, size:[{ob_s}, {ob_s}, 0.2, 0.01], color:[1 0 0], contact:1")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])
            
            elif tile == ROBOT:
                f = C.frame("ego").setRelativePosition([x_pos, y_pos, 0.0])
                f.setShape(ry.ST.ssCylinder, size=[.2, ob_s * .45, .02])
                
                # Handle cam0
                # Main config: attached to ego (for robot's perspective)
                try:
                    C.delFrame("cam0")
                except:
                    pass
                C.addFrame("cam0", "ego", f"Q:'t(0 0 0.5) d(180 1 0 0)' shape:camera, width:300, height:300")
                
                f_aux = C_aux.frame("ego").setRelativePosition([x_pos, y_pos, 0.0])
                # Aux config: Use fixed top-down camera attached to world (not ego) so it always sees entire scene
                # This ensures mask_object can always see all objects regardless of robot position
                try:
                    C_aux.delFrame("cam0")
                except:
                    pass
                # Top-down camera at height 10, looking down, attached to world (fixed position)
                C_aux.addFrame("cam0", "world", f"Q:'t(0 0 10) d(180 1 0 0)' shape:camera, width:300, height:300")
            
            elif tile == OBJECT:
                object_count += 1
                obj_name = f"obj{object_count}"
                joint_name = f"{obj_name}Joint"
                
                # Color palette
                object_colors = [
                    [0, 0, 1],      # Blue
                    [0, 1, 0],      # Green
                ]
                color_idx = min(object_count - 1, len(object_colors) - 1)
                obj_color = object_colors[color_idx]
                color_str = f"[{obj_color[0]} {obj_color[1]} {obj_color[2]}]"
                
                f = C.addFrame(joint_name, "world")
                f.setRelativePosition([x_pos, y_pos, 0.1])
                C.addFrame(obj_name, joint_name, 
                          f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str}, contact:1, joint:rigid, logical:{{movable_go}}")
                
                f_aux = C_aux.addFrame(joint_name, "world")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])
                C_aux.addFrame(obj_name, joint_name, 
                              f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str}, contact:1, logical:{{movable_go}}")
                C_aux.addFrame(f"{obj_name}_cam", obj_name, f"Q:'t(0 0 7) d(180 1 0 0)' shape:camera, width:300, height:300")
            
            elif tile == GOAL:
                goal_count += 1
                goal_name = f"goal{goal_count}"
                
                goal_colors = [
                    [0, 0, 1],      # Blue
                    [0, 1, 0],      # Green
                ]
                color_idx = min(goal_count - 1, len(goal_colors) - 1)
                goal_color = goal_colors[color_idx]
                color_str = f"[{goal_color[0]} {goal_color[1]} {goal_color[2]}]"
                color_str_transparent = f"[{goal_color[0]} {goal_color[1]} {goal_color[2]} .3]"
                
                f = C.addFrame(goal_name, "world", 
                             f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str_transparent}, contact:0, logical:{{goal}}")
                f.setRelativePosition([x_pos, y_pos, 0.1])
                
                f_aux = C_aux.addFrame(goal_name, "world", 
                                     f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:{color_str}, contact:0, logical:{{goal}}")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])
            
            elif tile == MOVABLE:
                # Name starts with "ob" so MOSeGMan's find_critical_objects can detect it
                movable_count += 1
                obj_name = f"ob_movable_{movable_count}"
                
                f = C.addFrame(f"{obj_name}Joint", "world")
                f.setRelativePosition([x_pos, y_pos, 0.1])
                C.addFrame(obj_name, f"{obj_name}Joint", 
                          f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[1 1 0], contact:1, joint:rigid, logical:{{movable_o}}")
                
                f_aux = C_aux.addFrame(f"{obj_name}Joint", "world")
                f_aux.setRelativePosition([x_pos, y_pos, 0.1])
                C_aux.addFrame(obj_name, f"{obj_name}Joint", 
                              f"shape:ssBox, size:[{ob_s*.6}, {ob_s*.6}, 0.2, 0.01], color:[1 1 0], contact:1, logical:{{movable_o}}")
                # Add camera for MOSeGMan's mask_object function (needed for object weight calculation)
                C_aux.addFrame(f"{obj_name}_cam", obj_name, f"Q:'t(0 0 7) d(180 1 0 0)' shape:camera, width:300, height:300")
    
    # Save configs
    new_C = C.write()
    new_C_aux = C_aux.write()
    
    with open(f"{output_dir}/test_movable.g", "w") as f:
        f.write(new_C)
    with open(f"{output_dir}/test_movable-aux.g", "w") as f:
        f.write(new_C_aux)
    
    print(f"Created test config:")
    print(f"  - Objects: {object_count} (right side)")
    print(f"  - Goals: {goal_count} (left side)")
    print(f"  - Movables: {movable_count} (blocking middle path)")
    print(f"  - Saved to: {output_dir}/test_movable.g")
    print(f"\nTo test:")
    print(f"  python segman_test.py --case_id test_movable --view --replay --verbose 1")
    
    return C, C_aux


if __name__ == "__main__":
    create_test_movable_config(size=17)
