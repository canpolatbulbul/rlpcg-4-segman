import robotic as ry
import os
from contextlib import contextmanager
import time
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import copy 
from collections import defaultdict, deque
import random
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt  

class Node():
    """
    Summary:
    -------------
    The Node used in Selective Guided Forward Search (SGFS).

    Inputs:
    -------------
    - C: The current configuration of the environment.
    - C_hm: The auxilary configuration.
    - objs: List of critical objects.
    - obj_weights: Dictionary of object weights for object relocation.
    - obj_scores: Dictionary of remaining object scores from potential scene score increase.
    - obj_pot_raw_score: The total potential raw score of the objects.
    - layer: The layer of the node in the search.
    - FS: The frame state list (mostly for visualization at the end).
    - moved_obj: The object that was moved to reach this node.
    - scene_score_raw_prev: The previous raw scene score before reaching this node.
    - reloc_o: Set of objects that have been relocated to reach this node.
    """

    def __init__(self, C:ry.Config, C_hm:ry.Config, objs:list, obj_weights:dict, obj_scores:dict, obj_pot_raw_score:float=0, layer:int=0, FS:list=[], moved_obj:str=None, scene_score_raw_prev:float=0, reloc_o:set = set()):
        self.C = ry.Config()
        self.C.addConfigurationCopy(C)
        self.C_hm = ry.Config()
        self.C_hm.addConfigurationCopy(C_hm) 
        self.objs = objs
        self.obj_weights = obj_weights
        self.obj_scores = obj_scores
        self.obj_pot_raw_score = obj_pot_raw_score
        self.layer = layer
        self.visit = 1
        self.FS = FS
        self.total_score = float("-inf")
        self.scene_score = 0
        self.scene_score_raw = 0
        self.scene_score_raw_prev = scene_score_raw_prev 
        self.id = random.randint(0, 1000000)
        self.pp_seq = 0
        self.moved_obj = moved_obj
        self.reloc_o = reloc_o

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.total_score == other.total_score
    
    def __lt__(self, other):
        if not isinstance(other, Node):
            return False
        return self.total_score > other.total_score
    
    def __str__(self):
        return "Node: layer={}, visit={}, pairs={}, total_score={}, id={}".format(self.layer, self.visit, self.objs, self.total_score, self.id)
    
    def __repr__(self):
        return self.__str__()

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

@contextmanager
def suppress_stdout():
    """
    To suppress output from RRT etc.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(devnull)
        os.close(original_stdout)
        os.close(original_stderr)

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

class SeGManv2():
    """
    Summary:
    -------------
    The Selective Guided Manipulation (SeGMan) class for object manipulation in cluttered environments.
    It generates a motion plan for a robot to manipulate a goal object while relocating obstacles as needed.

    Inputs:
    -------------
    - C: The current configuration of the environment.
    - C_hm: The auxilary configuration.
    - agent: The robot name.
    - obj_g_list: List of goal objects in the configuration.
    - all_objs: List of all movable objects in the configuration.
    - obs_list: List of movable obstacles in the configuration.
    - verbose: Verbosity level for logging.
    """
    def __init__(self, C:ry.Config, C_hm:ry.Config, agent:str, obj_g_list:list, all_objs:list, obs_list:list, verbose:int):
        self.C = ry.Config()
        self.C.addConfigurationCopy(C)
        if C_hm is not None:
            self.C_hm = ry.Config()
            self.C_hm.addConfigurationCopy(C_hm)
        else:
            self.C_hm = None
        self.agent = agent
        self.obj_g_list = copy.deepcopy(obj_g_list)
        self.all_objs = copy.deepcopy(all_objs)
        self.obs_list = copy.deepcopy(obs_list)
        self.verbose = verbose
        self.FS = []
        self.obj_msks = {}
        self.verbose = verbose
        self.ego_size = None

    def solve(self, obj, placed_objs):
        """
        Summary:
        -------------
        Function to solve the manipulation problem for a given object.
        It finds a feasible pick path, checks for obstacles, and generates a manipulation plan.

        Inputs:
        -------------
        - obj: The object to be placed.
        - goal: The goal position for the object.

        Outputs:
        -------------
        - Returns True if the object is successfully placed, False otherwise. The motion plan frame states are stored in self.FS.
        -------------
        """
        
        found = False
        bttlnck = False # Flag to push the planner to generate an alternative placement object trajectory
        max_iter = 2
        iter = -1
        path_found = False
        is_picked = False
        reloc_o = set()
        place_P = []
        tot_pp_seq = 0
        self.obj = obj
        self.goal = self.C.frame("goal"+obj[3:]).getPosition()[0:2]
        if self.verbose > 0:
            print("Trying to move", obj)

        while not found and iter < max_iter:

            iter += 1

            # Find a feasible pick configuration and path
            pick_FS = []
            f_pick = self.find_pick_path(self.C, self.agent, self.obj, pick_FS, self.verbose, K=4, N=2, step_size=0.03)

            # If it is not possible to pick obj, check if there is an obstacle that can be removed
            if not f_pick: 
                if len(self.obs_list) > 0:
                    # Attempt to remove the obstacles
                    f_pick_ro, _, pp_seq, rel_o = self.remove_obstacle(placed_objs, type=0)
                    if f_pick_ro:
                        reloc_o.update(rel_o)
                        tot_pp_seq += pp_seq
                        is_picked = False
                        self.C.setFrameState(self.FS[-1])
                    else:
                        print("Solution not found!")
                        return False, 0, list(reloc_o)
                else:
                    print("Solution not found!")
                    return False, 0, list(reloc_o)
            else:
                is_picked = True

            step_size = 0.03
            slicesPerPhase = 40
            step_div = 2
            l = -1
            L = 3
            while l < L:
                l += 1
                # Generate trajectory for the object to its goal position
                if l == L-1:
                    # Make the object an agent to run RRT for it
                    self.C2 = self.make_agent(self.C, self.obj, self.agent, place_P, bttlnck)
                else:
                    self.C2 = self.make_agent(self.C, self.obj, self.agent, place_P, False)
                    
                f_place, place_P = self.run_rrt(self.C2, self.goal, self.verbose, N=3, step_size=step_size, isOpt=True)
                if l == L-1:
                    self.C2.delFrame("tmp_blck")

                # If there is no feasible trajectory for object, check if there is an obstacle that can be removed
                if not f_place: 
                    # Attempt to remove the obstacles
                    if len(self.obs_list) > 0:
                        f_place_ro, place_P, pp_seq, rel_o = self.remove_obstacle(placed_objs, type=1)
                        if f_place_ro:
                            reloc_o.update(rel_o)
                            tot_pp_seq += pp_seq
                            self.C.setFrameState(self.FS[-1])
                            is_picked = False
                            break
                        else:
                            print("Solution not found!")
                            return False, 0, list(reloc_o)
                    else:
                        print("Solution not found!")
                        return False, 0, list(reloc_o)

                # Check if the object is currently picked
                if is_picked: 
                    self.C.setFrameState(pick_FS[-1])

                # Generate the manipulation plan for the object by adaptively selecting trajectory points
                path_found, seq_info, bttlnck, _, _, pp_seq = self.gen_adaptive_manip_plan(self.C, place_P, self.agent, self.obj, 0, cfc_lim=3,  K=3, slicesPerPhase=slicesPerPhase, step_div=step_div)
                if path_found:
                    self.FS.extend(pick_FS)
                    # Refine the selected subgoals
                    pp_seq = self.refine_manip_plan(self.C, seq_info, self.agent, self.obj, slicesPerPhase, self.FS, pp_seq=pp_seq) 
                    tot_pp_seq += pp_seq   
                    self.C.setFrameState(self.FS[-1])
                    #print("Solution found for:", self.obj)
                    return True, tot_pp_seq, list(reloc_o)
                else:
                    step_size -= 0.005 * l
                    slicesPerPhase += l*10

        if self.verbose > 0:
            print("Max iterations reached. No solution found for:", self.obj)            
        return False, 0, list(reloc_o)

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def remove_obstacle(self, placed_objs: list, type: int):

        """
        Summary:
        -------------
        Function to remove obstacles for picking or placing objects based on the type specified.
        
        Inputs:
        -------------
        - type: Integer indicating the type of operation (0 for pick, 1 for place).

        Outputs:
        -------------
        - Returns a tuple (feasible, path, cost) where:
            - feasible: Boolean indicating if the operation is feasible.

        -------------
        """
        for o in self.all_objs:
            self.C_hm.frame(o).setPosition(self.C.frame(o).getPosition())
            self.C_hm.frame(self.agent).setPosition([*self.C.frame(self.agent).getPosition()[:2],0])

        self.type = type

        if self.verbose > 0:
            if type == 0:
                print("Removing obstacle for pick")
            else:   
                print("Removing obstacle for place")
        
        # Add alternative set
        blocking_objs, path = self.find_critical_objects(type, self.all_objs, placed_objs, verbose=self.verbose)

        if blocking_objs is None:
            return False, None, 0, []
        
        obj_weight, obj_score_raw, C_hm, score_init, tot_raw_score = self.calc_object_weights(blocking_objs, path)
        root = Node(self.C, C_hm=C_hm, objs=blocking_objs, obj_weights=obj_weight, obj_scores=obj_score_raw, obj_pot_raw_score=tot_raw_score)
        root.scene_score_raw_prev = score_init
        self.node_eval(root)
        f, p, c, reloc_o = self.selective_guided_forward_search(root)
        
        if f:
            return f, p, c, reloc_o
        
        else:
            rem_objs = [o for o in self.all_objs if o not in blocking_objs]
            blocking_objs, path = self.find_critical_objects(type, rem_objs, placed_objs, verbose=0)

            if blocking_objs is None:
                return False, None, 0, []
            
            obj_weight, obj_score_raw, C_hm, score_init, tot_raw_score = self.calc_object_weights(blocking_objs, path)
            root = Node(self.C, C_hm=C_hm, objs=blocking_objs, obj_weights=obj_weight, obj_scores=obj_score_raw, obj_pot_raw_score=tot_raw_score)
            root.scene_score_raw_prev = score_init
            self.node_eval(root)
            f, p, c, reloc_o = self.selective_guided_forward_search(root)
            return f, p, c, reloc_o

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def selective_guided_forward_search(self, root_node:Node):
        """
        Summary:
        -------------
        Function to perform a selective guided forward search to find a feasible path for picking or placing an object.
        
        Inputs:
        -------------
        - root_node: The initial node with the current configuration and critical objects.

        Outputs:
        -------------
        - Returns a tuple (feasible, trajectory, pick-and-place count, relocated objs) where:
            - feasible: Boolean indicating if a feasible path was found.
            - trajectory: The trajectory path if found, None otherwise.
            - pick-and-place count: The number of pick-and-place operations performed.
            - relocated objs: Set of objects that were relocated during the search.
        -------------
        """

        prev_node = None
        N = [root_node]
        max_iter = len(self.obj_g_list)+1
        idx_i = 0
        node_stuck_c = 0
        node_stuck_lim = max(2, len(self.obj_g_list)/4)
        col_dict = defaultdict(int)

        while idx_i < max_iter and node_stuck_c < node_stuck_lim:
            idx_i+=1
            
            # Select the best node
            node = self.select_node(N, prev_node, col_dict)
            if self.verbose > 0:
                print("Trying to relocate", node.objs)

            # Check if the node is feasible for picking or placing the object
            if self.type==0 and prev_node != None and node.id != prev_node.id:
                if self.verbose > 0:
                    print("Checking if picking configuration is feasible")

                f_pick = self.find_pick_path(node.C, self.agent, self.obj, node.FS, self.verbose, K=4, N=3)

                if self.verbose > 0:
                    print("Picking path found: ", f_pick)

                if f_pick:
                    self.FS.extend(node.FS)
                    return True, None, node.pp_seq, node.reloc_o
                
            elif self.type==1 and prev_node != None and node.id != prev_node.id:
                if self.verbose > 0:
                    print("Checking if placement configuration is feasible")

                C2 = self.make_agent(node.C, self.obj, self.agent)
                f_place, path = self.run_rrt(C2, self.goal, self.verbose, N=4, step_size=0.03, isOpt=True, max_iter=2000, delta_iter=2000)

                if self.verbose > 0:
                    print("Placement path found: ", f_place)

                if f_place:
                    self.FS.extend(node.FS)
                    return True, path, node.pp_seq, node.reloc_o    
            
            # If there is no improvement in the scene score, the search is stuck
            if prev_node != None and node.scene_score <= prev_node.scene_score :
                node_stuck_c += 1
            else:
                node_stuck_c = 0

            prev_node = node
            any_reach = False

            # Try to relocate all reachable critical objects in the node
            for o in node.obj_weights.keys():
                if o != node.moved_obj:
                    is_reach, reach_P = self.is_reachable(node, o)

                    # Skip the object if it is not reachable
                    if not is_reach:  
                        continue
                    

                any_reach = True
            
                if self.verbose > 0:
                    print(f"Object {o} is REACHABLE")

                # For each reachable object, generate subnodes and return the highest scoring 3 nodes
                self.generate_new_nodes(node, o, N, P=reach_P, sample_count=3, col_dict=col_dict)

            # If there is no reachable object in the current node, remove it from the list
            if not any_reach:
                if self.verbose > 0:
                    print("Node REMOVED:", node.objs)
                N.remove(node)
                self.prev_node = None
                if len(N) == 0:
                    if self.verbose > 0:
                        print("NO remaining nodes!")
                    return False, None, 0, set()

        return False, None, 0, set()

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def generate_obj_comb(self, ob_list:list):
        """
        Summary:
        -------------
        Function to generate all combinations of objects.
        
        Inputs:
        -------------
        - ob_list: List of objects to generate combinations from.

        Outputs:
        -------------
        - Returns a list of all combinations of objects.
        -------------
        """
        obj_comb = []
        for r in range(1, len(ob_list) + 1):
            obj_comb.extend(combinations(ob_list, r))

        obj_comb = [list(item) for item in obj_comb]
        return obj_comb

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def find_critical_objects(self, type:int, obj_list:list, placed_objs:list, verbose:int=0):
        """
        Summary:
        -------------
        Function to find the minimal set of critical objects which blocks the task trajectory.

        Inputs:
        -------------
        - type: Integer indicating the type of task (0 for pick, 1 for place).
        - obj_list: List of movable objects to be considered in the search.
        - placed_objs: List of already placed objects.
        - verbose: Verbosity level for logging.
        
        Output:
        -------------
        - Returns a tuple (blocking_objs, path) where:
            - blocking_objs: List of critical objects that block the task trajectory, None if no blocking objects are found.
            - path: The task trajectory path if found, None otherwise.
        """
        if self.verbose > 0:
            print("Finding collision pairs")

        Ct = ry.Config()
        if type == 0:
            obj_goal = self.C.frame(self.obj).getPosition()[0:2]
            Ct.addConfigurationCopy(self.C)
            shape = self.C.frame(self.agent).getShapeType()
            size  = self.C.frame(self.agent).getSize()
            size = [s * 1.2 for s in size]
            quat = self.C.frame(self.agent).getQuaternion()
            z = self.C.frame(self.agent).getPosition()[2]
            color = self.C.frame(self.agent).getAttributes()["color"]
        else:
            obj_goal = self.goal
            Ct.addConfigurationCopy(self.C2)
            shape = self.C.frame(self.obj).getShapeType()
            size  = self.C.frame(self.obj).getSize()
            size = [s * 1.2 for s in size]
            quat = self.C.frame(self.obj).getQuaternion()
            z = self.C.frame(self.obj).getPosition()[2]
            color = self.C.frame(self.obj).getAttributes()["color"]

        # Disable contact for all objects except the target object if type is 1 (place)
        for o in obj_list:
            if not (type == 1 and o == self.obj):    
                if o not in placed_objs: 
                    Ct.frame(o).setContact(0)

        # Generate task trajectory
        f, path = self.run_rrt(Ct, obj_goal, verbose, N=2, step_size=0.03, isOpt=False, delta_iter=3000)

        # Also disable collision for the placed objects
        if not f:
            for o in obj_list:
                if not (type == 1 and o == self.obj):  
                    Ct.frame(o).setContact(0)

            f, path = self.run_rrt(Ct, obj_goal, verbose, N=2, step_size=0.03, isOpt=False, delta_iter=3000)
        

        if f:

            # Reennable contact for all objects except the target object if type is 1 (place) to check collisions along the trajectory
            for o in obj_list:
                if not o in placed_objs:
                    Ct.frame(o).setContact(1)
                else:
                    Ct.frame(o).setContact(0)

            Ct2 = ry.Config()
            Ct2.addConfigurationCopy(Ct)
            Ct2.frame(self.obj).setContact(0)

            # Add object clones along the trajectory to check for collisions
            for i in range(0, len(path), 3):
                if len(path) - i < 3:
                    pi = path[-1]
                else:
                    pi = path[i]

                clone = Ct2.addFrame("tmp" + str(i))
                clone.setColor([*color, 0.5])
                clone.setShape(shape, size)
                clone.setQuaternion(quat)
                clone.setPosition([*pi, z])
                clone.setContact(1)
            
            # Obtain colliding objects with the generated trajectory
            blocking_objs = set()
            col = Ct2.getCollisions(0)
            for o1, o2, val in col:
                if abs(val) < 1e-3: continue
                if (o1.startswith("tmp") and o2.startswith("ob")) or (o1.startswith("ob") and o2.startswith("tmp")):
                    if o1.startswith("tmp"):
                        blocking_objs.add(o2)
                    else:
                        blocking_objs.add(o1)

            if verbose > 0:
                print("Blocking objects found:", blocking_objs)

            if len(blocking_objs) > 1:
                
                # If the number of blocking objects is small, check all combinations to find the minimal set
                if len(blocking_objs) <=4:
                    obs_combs = self.generate_obj_comb(list(blocking_objs))
                    for comb in obs_combs:
                        Ct3 = ry.Config()
                        Ct3.addConfigurationCopy(Ct)

                        for o in obj_list:
                            Ct3.frame(o).setContact(1)
        
                        if type == 0:
                            Ct3.frame(self.obj).setContact(0) 

                        for o in comb:
                            Ct3.frame(o).setContact(0)
    
                        f, path = self.run_rrt(Ct3, obj_goal, verbose, N=2, step_size=0.03, isOpt=True)

                        if f:
                            if verbose > 0:
                                print("Blocking objects found:", comb)
                            return comb, path
                else:
                    dist_l = []
                    rp = self.C.frame(self.agent).getPosition()[:2]
                    for o in blocking_objs:
                        dist_l.append(np.linalg.norm(rp - self.C.frame(o).getPosition()[:2]))
                    
                    # Find the top 4 closest object to robot in dist_l 
                    top_4_closest = np.argsort(dist_l)[:4]
                    comb = [list(blocking_objs)[i] for i in top_4_closest]
                    return comb, path

            elif len(blocking_objs) > 0:
                return list(blocking_objs), path
                
        if verbose > 1:
            Ct2.view(True)
            Ct2.view_close()
            print("Blocking objects NOT found")

        return None, None
                
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def calc_object_weights(self, blocking_objs:list, path:list):
        """
        Summary:
        -------------
        Function to assign weights for each critical object (0-1) based on their impact on the scene score.
        The weights are used to determine how many relocation points will be proposed for each object.

        Inputs:
        -------------
        - blocking_objs: List of critical objects that block the task trajectory.
        - path: The task trajectory.

        Outputs:
        -------------
        - Returns a tuple (obj_weight, obj_score_raw, C_hm, score_init, total_score) where:
            - obj_weight: Dictionary of object weights for object relocation.
            - obj_score_raw: Dictionary of remaining object scores from potential scene score increase.
            - C_hm: The auxilary configuration with object clones along the trajectory.
            - score_init: The initial scene score before any object relocation.
            - total_score: The total potential raw score of the objects.
        """
        ry.params_clear()
        ry.params_add({"Render/useShadow":False})

        Ct_hm = ry.Config()
        Ct_hm.addConfigurationCopy(self.C_hm)

        if self.type == 0:
            shape = self.C.frame(self.agent).getShapeType()
            size = self.C.frame(self.agent).getSize() * 1.3
            z = self.C.frame("floor").getPosition()[2]
        else:
            shape = Ct_hm.frame(self.obj).getShapeType()
            size = Ct_hm.frame(self.obj).getSize() * 1.3
            z = self.C.frame("floor").getPosition()[2]

        # Add object clones along the trajectory to the auxilary configuration (for GOM (Global Occupancy Matrix) and LOM (Local Occupancy Matrix) calculation)
        for i, p in enumerate(path):
                clone = Ct_hm.addFrame("clone_"+str(i), "egoJoint")
                clone.setShape(shape, size)
                clone.setPosition([*p, z])
                clone.setColor([0, 0, 1])

        # Compute the potential increase in scene score for each object
        camera_view = ry.CameraView(Ct_hm)
        cam = Ct_hm.getFrame("cam0")
        camera_view.setCamera(cam)
        img_init, _ = camera_view.computeImageAndDepth(Ct_hm)
        img_init = self.scene_adjust(img_init)
        score_init = sum(img_init.flatten())
        obj_score = {}
        for obj in blocking_objs:
            obj_msk = self.mask_object(Ct_hm, camera_view, obj, False)
            Ct_hm.frame(obj).setColor([1, 1, 1, 0])
            img_obj, _ = camera_view.computeImageAndDepth(Ct_hm)
            img_obj = self.scene_adjust(img_obj)
            score_obj = sum(img_obj.flatten()) - sum(obj_msk.flatten())*80
            scene_delta = score_obj - score_init
            obj_score[obj] = scene_delta if scene_delta > 0 else 0
            Ct_hm.frame(obj).setColor([1, 0, 0, 1])
        
        # Find the robot size to be used in reachability matrix generation (wavelet-propagation)
        if self.ego_size is None:
            Ct = ry.Config()
            Ct.addConfigurationCopy(self.C)
            camera_view2 = ry.CameraView(Ct)
            cam2 = Ct.getFrame("cam0")
            camera_view2.setCamera(cam2)
            img_init, _ = camera_view2.computeImageAndDepth(Ct)
            ego_mask = self.mask_object(Ct, camera_view2, self.agent, False)
            self.ego_size = ego_mask.shape

        # Normalize the object scores to sum up to 1
        total_score = sum(obj_score.values())

        if total_score > 0:
            obj_weight = {obj: score / total_score for obj, score in obj_score.items()}
        else:
            obj_weight = {obj: 1 / len(obj_score) for obj in obj_score}

        if self.verbose > 1:
            print("Object Weights: ", obj_weight)
            print("Object Scores: ", obj_score)

        return obj_weight, obj_score, Ct_hm, score_init, total_score

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def select_node(self, N:list, prev_node:Node, col_dict:dict):
        """
        Summary:
        -------------
        Function to select the best node from the list of nodes based on their scores and exploration-exploitation trade-off.
        
        Inputs:
        -------------
        - N: List of Node objects to select from.
        - prev_node: The previously selected node.
        - col_dict: Dictionary to keep track of colliding objects detected during new node generation.

        Outputs:
        -------------
        - best_node: The Node object with the highest score.
        -------------
        """
        
        if self.verbose > 0:
            print("Selecting the node")

        best_node = None
        if prev_node is not None:
            prev_node.visit += 1
            self.node_score(prev_node)

        # Select the best node
        best_node = max(N, key=lambda n: n.total_score)
        
        # If the best node has been already visited, include the secondary obstacles to the critical object set of the node
        if prev_node != None and prev_node.scene_score >= best_node.scene_score and len(col_dict) > 0:
            red_obj = max(col_dict, key=col_dict.get)
            if red_obj not in best_node.objs:
                col_dict.pop(red_obj) 
                best_node.objs.append(red_obj)
                
                # The weight of the new obstacle is proportional to its size
                camera_view = ry.CameraView(best_node.C_hm)
                cam = best_node.C_hm.getFrame(red_obj + "_cam")
                camera_view.setCamera(cam)
                msk = self.mask_object(best_node.C_hm, camera_view, red_obj, True)
                sum_msk = sum(msk.flatten())*200

                shape = self.C.frame(red_obj).getShapeType()
                size = self.C.frame(red_obj).getSize()
                size = [s * 0.6 for s in size]  # Scale down the size
                pos = self.C.frame(red_obj).getPosition()
                fr = best_node.C_hm.addFrame(str(red_obj)+"_b", "world", "color:[0 0 0], contact:1")
                fr.setShape(shape, size)
                fr.setPosition(pos)
                fr.setColor([0, 0, 255])
                
                best_node.obj_scores[red_obj] = sum_msk * 0.6
                best_node.obj_pot_raw_score = sum(best_node.obj_scores.values())
                best_node.obj_weights = {obj: score / best_node.obj_pot_raw_score for obj, score in best_node.obj_scores.items()}
                
                if self.verbose > 0:
                    print(f"Redundant object {red_obj} added to the node {best_node.id}")
                    print("Selected Node: " , best_node)
                    print("Object Weights: ", best_node.obj_weights)
                    if self.verbose > 1:
                        best_node.C.view(True, str(best_node))
                        best_node.C.view_close()

                return best_node
        
        # Update the object weights based on the score change after evaluating the node
        if best_node.moved_obj is not None:
            score_delta = best_node.scene_score_raw - best_node.scene_score_raw_prev
            best_node.scene_score_raw_prev = best_node.scene_score_raw
            # Distribute gain to insignificant objects
            if score_delta > 0:
                denom = sum(1 for v in best_node.obj_scores.values() if v == 0)
                if denom > 0:
                    best_node.obj_scores = {k: (v + score_delta/denom if v == 0 else v) for k, v in best_node.obj_scores.items()}

            best_node.obj_scores[best_node.moved_obj] -= score_delta
            best_node.obj_scores[best_node.moved_obj] = max(best_node.obj_scores[best_node.moved_obj], 0.001)
            best_node.obj_pot_raw_score = sum(best_node.obj_scores.values())
            
            # Update the node weights
            best_node.obj_weights = {obj: score / best_node.obj_pot_raw_score for obj, score in best_node.obj_scores.items()}

        if self.verbose > 0:
            print("Selected Node: " , best_node)
            print("Object Weights: ", best_node.obj_weights)
            if self.verbose > 1:
                best_node.C.view(True, str(best_node))
                best_node.C.view_close()
        
        return best_node
    
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def node_eval(self, node:Node, verbose:int=0):
        """
        Summary:
        -------------
        Function to evaluate the node by updating its configuration and calculating its score.
        
        Inputs:
        -------------
        - node: The Node object to be evaluated.
        - obj: The object to be evaluated.
        - verbose: Verbosity level for logging.

        Outputs:
        -------------
        - Returns None. The node's configuration and score are updated in place.
        -------------
        """
        
        self.update_chm(node)
    
        # Score the scene
        node.scene_score, node.scene_score_raw = self.scene_score(node.C_hm, "cam0", verbose)

        # Score the node
        self.node_score(node)
                
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def node_score(self, node:Node):
        """
        Summary:
        -------------
        Function to calculate the score of the node based on its exploitation and exploration values.
        
        Inputs:
        -------------
        - node: The Node object to be scored.

        Outputs:
        -------------
        - Returns None. The node's total score is updated in place.
        -------------
        """

        exploitation = node.scene_score
        exploration  = 5 * 1/math.sqrt(node.visit + 1)
        node.total_score = exploitation + exploration
                
        if self.verbose > 0:
            print("Scored Node: ", node)
            print(f"Scene Score:  {node.scene_score}, Exploitation: {exploitation}, Exploration: {exploration}, Total Score: {node.total_score}")

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def make_agent(self, C:ry.Config, obj:str, agent:str="", P:list=[], bttlnk:bool=False):
        """
        Summary:
        -------------
        Function to remove the agent from the configuration and add a new agent frame with the specified object.
        
        Inputs:
        -------------
        - C: The configuration of the environment.
        - obj: The object to be designated as the agent.
        - agent: The current agent in the configuration.
        - P: The picking trajectory for the agent, if available.
        - bttlnk: Boolean flag to indicate if a blocking object should be added for alternative path generation.

        Outputs:
        -------------
        - Returns a new configuration with the agent frame removed and the specified object added as an agent.
        -------------
        """

        C2 = ry.Config()
        C2.addConfigurationCopy(C)

        C2.delFrame(agent)
        C2.delFrame(agent+"Joint")
        
        obj_size   = C.frame(obj).getSize()
        obj_shape  = C.frame(obj).getShapeType()
        pose = C.frame(obj).getPose()     
        color = C.frame(obj).getAttributes()["color"]
        z = C.frame(obj+"Joint").getPosition()[2]
        C2.delFrame(obj)
        C2.delFrame(obj+"Joint")
        obj_J = C2.addFrame(obj+"Joint", "world")
        obj_J.setPosition([0, 0, z])
        obj_f = C2.addFrame(obj, obj+"Joint")
        obj_f.setContact(1)
        obj_f.setShape(obj_shape, size=obj_size)
        obj_f.setColor(color)
        obj_f.setJoint(ry.JT.transXY, limits=[-4, 4, -4, 4])
        C2.setJointState(pose[0:2])
        obj_f.setQuaternion(pose[3:7])

        # Add the blocking object to the scene for alternative RRT path generation
        if bttlnk and len(P)>0:
            fr = C2.addFrame("tmp_blck", "world", "color:[0 0 0], contact:1")
            fr.setShape(obj_shape, size=obj_size)
            fr.setPosition([*P[int(len(P)*0.5)][0:2], z])
            fr.setColor([0, 0, 0])
            fr.setContact(1)

        # Increase the size of the object if the agent is larger than the object so that if an object can fit through a gap, the agent can also fit through it
        if agent != "":
            agent_size = C.frame(agent).getSize()
            obj_size   = C.frame(obj).getSize()
            if max(obj_size[0], obj_size[1]) < agent_size[0]:
                obj_f.setShape(ry.ST.ssBox, size=[agent_size[0]*1, agent_size[0]*1, .2, .02])

        return C2
    
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def generate_new_nodes(self, node:Node, o:str, N:list, P:list=[], sample_count:int = 3, col_dict:dict={}):
        """
        Summary:
        -------------
        Function to generate subnodes for the given node by finding alternative relocation positions for the specified object.
        
        Inputs:
        -------------
        - node: The Node object for which subnodes are to be generated.
        - o: The object for which subnodes are to be generated.
        - N: List of Node objects to which the generated subnodes will be added.
        - P: The picking trajectory for the agent, if available.
        - sample_count: Number of subnodes to generate.
        - col_dict: Dictionary to keep track of colliding objects detected during new node generation.

        Outputs:
        -------------
        - None, the generated subnodes are added to the list N.
        -------------
        """

        if self.verbose > 0:
            print("Generate subgoals")

        Ct = ry.Config()
        Ct.addConfigurationCopy(node.C)

        # Add the picking motion plan to the frame state list (generated during reachability check)
        base_state = Ct.getFrameState()
        fs_base = copy.deepcopy(node.FS)
        generated_nodes = []
        if len(P) > 0:
            fs_base.extend(P)
            base_state = P[-1]

        # Calculate how many points to sample based on the weight of the object weight and number of objects
        amount = int((node.obj_weights[o] / sum(node.obj_weights.values())) * 5 * math.ceil(math.sqrt(len(node.objs))))
        amount = max(0, amount)

        if amount == 0:
            return
        
        # Generate object relocation trajectories
        P = self.generate_reloc_pts(node, o, amount, self.verbose)

        if len(P) == 0:
            return 
        
        for path in P:
            Ct.setFrameState(base_state)
            is_feas, seq_info, _, path_frames, komo, pp_seq = self.gen_adaptive_manip_plan(Ct, path, self.agent, o, self.verbose, cfc_lim=3,  K=2, step_div=10, min_step=5)
            if len(seq_info) > 0:
                fs = copy.deepcopy(fs_base)
                fs.extend(path_frames)
                Ct.setFrameState(path_frames[-1])
                reloc_o = copy.deepcopy(node.reloc_o)
                reloc_o.add(o)
                new_node = Node(C=Ct, C_hm=node.C_hm, objs=copy.deepcopy(node.objs), obj_weights=copy.deepcopy(node.obj_weights), obj_scores=copy.deepcopy(node.obj_scores), obj_pot_raw_score=node.obj_pot_raw_score, layer=node.layer+1, FS=copy.deepcopy(fs), moved_obj=o, scene_score_raw_prev=node.scene_score_raw_prev, reloc_o=reloc_o)
                new_node.pp_seq += pp_seq
                self.node_eval(new_node, verbose=self.verbose)
                generated_nodes.append(new_node)

            # Find the colliding objects 
            if not is_feas:
                for i in range(60):
                    col_info = komo.info_sliceCollisions(i, 10)
                    if col_info != "":
                        col_objs = [line.split(":")[0].split("-") for line in col_info.splitlines() if ":" in line]
                        for objs in col_objs:
                            if objs[0] != o and self.agent not in objs and objs[0] in self.all_objs and objs[0] not in node.objs and objs[0] != self.obj:
                                col_dict[objs[0]] += 1
                            elif objs[1] != o and self.agent not in objs and objs[1] in self.all_objs and objs[1] not in node.objs and objs[1] != self.obj:
                                col_dict[objs[1]] += 1
                                
        # Sort the generated nodes based on their total score and select the top sample_count nodes
        generated_nodes = sorted(generated_nodes, key=lambda n: n.total_score, reverse=True)
        N.extend(generated_nodes[:min(sample_count, len(generated_nodes))])

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def update_chm(self, node:Node):
        """
        Summary:
        -------------
        Function to update the configuration heatmap (C_hm) of the node by setting the positions of the objects in the cluster.
        
        Inputs:
        -------------
        - node: The Node object whose configuration heatmap is to be updated.

        Outputs:
        -------------
        - None, the node's C_hm is updated in place.
        -------------
        """

        for o in node.objs:
            node.C_hm.frame(o).setPosition(node.C.frame(o).getPosition())
        z = node.C.frame(o).getPosition()[2]
        node.C_hm.frame(self.agent).setPosition([*node.C.frame(self.agent).getPosition()[:2],z])

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
    def mask_object(self, C:ry.Config, camera_view:ry.CameraView, obj:str, is_save:bool=True):
        """
        Summary:
        -------------
        Function to generate object mask from the global view.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - camera_view: The camera view object for rendering.
        - obj: The object for which the mask is to be generated.
        - is_save: Boolean flag to indicate if the generated mask should be saved for future use to the obj_msks dictionary.

        Outputs:
        -------------
        - Returns the cropped mask of the specified object.
        """

        if obj not in self.obj_msks.keys():

            # Identify the object from it's unique color (turquoise)
            col = C.frame(obj).getAttributes()["color"]
            C.frame(obj).setColor([0, 255, 255])
            obj_msk, _ = camera_view.computeImageAndDepth(C)
            R = obj_msk[:, :, 0]
            G = obj_msk[:, :, 1]
            B = obj_msk[:, :, 2]
            turquoise_mask = (G > 200) & (B > 200) & (R < 50)
            obj_msk[~turquoise_mask] = 0
            coords = np.column_stack(np.where(turquoise_mask > 0))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            # Add some padding to the mask
            cropped_mask = obj_msk[y_min-2:y_max+3, x_min-2:x_max+3]
            R2 = cropped_mask[:, :, 0]
            G2 = cropped_mask[:, :, 1]
            B2 = cropped_mask[:, :, 2]
            turquoise_mask2 = (G2 > 200) & (B2 > 200) & (R2 < 50)
            cropped_mask[turquoise_mask2] = [0, 1, 0]
            if is_save:
                self.obj_msks[obj] = cropped_mask
            C.frame(obj).setColor(col)
        else:
            cropped_mask = self.obj_msks[obj]
        return cropped_mask

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def generate_reloc_pts(self, node:Node, obj:str, amount:int=50, verbose:int=0):
        """
        Summary:
        -------------
        Function to generate potential relocation points for the specified object using camera view and depth image processing.

        Inputs:
        -------------
        - node: The Node object containing the current configuration.
        - obj: The object for which relocation points are to be generated.
        - amount: The number of relocation points to generate.
        - verbose: Verbosity level for logging.

        Outputs:
        -------------
        - Returns a list of relocation trajectories.
        """

        ry.params_clear()
        ry.params_add({"Render/useShadow":False})

        conv_thresh = 0.9
        edt_thresh  = 75

        # Get the Local Occupancy Matrix (LOM) from the camera view
        camera_view = ry.CameraView(node.C_hm)
        cam = node.C_hm.getFrame(obj+"_cam")
        camera_view.setCamera(cam)
        img, depth = camera_view.computeImageAndDepth(node.C_hm)
        img = np.asarray(img)
        img_c = copy.deepcopy(img)
        img = img / 255.0

        # Get the obj mask
        obj_mask = self.mask_object(node.C_hm, camera_view, obj)
        obj_maskf = np.flip(obj_mask[:, :, 1])
        tot = sum(obj_mask[:, :, 1].flatten())
        
        conv_map = convolve2d(img[:, :, 1], obj_maskf, mode='same', boundary="fill", fillvalue=0)
        conv_c = conv_map / tot  # Normalize the convolution result
        conv_c[conv_c < conv_thresh] = 0 # Threshold the convolution map

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        yellow = (R > 0.5) & (G > 0.5) & (B < 0.5)
        green  = (R < 0.5) & (G > 0.5) & (B < 0.5)
        blue   = (R < 0.5) & (G < 0.5) & (B > 0.5)
        free_space = yellow | green | blue

        # Compute the Euclidean Distance Transform (EDT) of the free space
        edt_raw = distance_transform_edt(free_space)

        # Threshold the EDT map to remove cluttered placement locations
        tr = np.percentile(edt_raw, edt_thresh)
        img_c = self.scene_adjust(img_c)
        conv_c[edt_raw < tr] = 0

        # Generate relocation points
        depth[conv_c == 0] = 0
        pts = ry.depthImage2PointCloud(depth, camera_view.getFxycxy())
        pts = self.cam_to_target(pts.reshape(-1, 3), node.C_hm.getFrame("world"), cam)
        

        k = min(max(math.ceil(len(pts) * 0.03), amount), len(pts))
        pts = np.asarray(pts)  # ensure numpy array
        pts = pts[np.random.choice(len(pts), size=k, replace=False)]


        # For each relocation point generate placement trajectories
        Ct = ry.Config()
        Ct.addConfigurationCopy(node.C)
        count = 0
        idx = 0
        P = []
        Ct2 = self.make_agent(Ct, obj, self.agent)
        while count < amount and idx < min(len(pts), 20):
            p = pts[idx]
            f_place, place_P = self.run_rrt(Ct2, p[:2], 0, N=2, isOpt=True)
            if f_place:
                count += 1
                P.append(place_P)
            idx += 1
        
        if len(P) == 0:
            for o in self.all_objs:
                if o != obj:
                    Ct.frame(o).setContact(0)
            Ct2 = self.make_agent(Ct, obj, self.agent)
            for p in pts:
                f_place, place_P = self.run_rrt(Ct2, p[:2], 0, N=2, isOpt=True)
                if f_place:
                    P.append(place_P)
                if len(P) >= amount:
                    break

        if(verbose > 1):
            # Plot the camera image 
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img_c, cmap="viridis")
            plt.title("Camera Image")
            plt.axis('off')
            
            # Plot msk 
            plt.subplot(1, 3, 2)
            plt.imshow(edt_raw, cmap="RdYlGn")
            plt.title("EDT Map")
            plt.axis('off')

            # Plot msk 
            plt.subplot(1, 3, 3)
            plt.imshow(conv_c, cmap="gray")
            plt.title("Convolution Map")
            plt.axis('off')
            plt.show()

            C2 = ry.Config()
            C2.addConfigurationCopy(node.C)
            C2.getFrame("world").setPointCloud(pts, [0,0,0])
            C2.view(True,"1")

        return P

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def is_reachable(self, node:Node, o:str):
        """
        Summary:
        -------------
        Function to check if the specified object is reachable from the current node's configuration.
        
        Inputs:
        -------------
        - node: The Node object containing the current configuration and reachable objects.
        - o: The object to check for reachability.

        Outputs:
        -------------
        - Returns a tuple (is_reachable, P) where:
            - is_reachable: Boolean indicating if the object is reachable.
            - P: List of frames that for the picking motion plan.
        -------------
        """

        if self.verbose > 0:
            print(f"Checking if object {o} is reachable")

        P = []

        # If the object is already reachable, return True
        if o == node.moved_obj:
            return True, P
        
        is_reachable = self.find_pick_path(node.C, self.agent, o, FS=P, verbose=self.verbose, K=4, N=2) 
        return is_reachable, P

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def find_pick_path(self, C:ry.Config, agent:str, obj:str, FS:list, verbose: int, K:int=5, N:int=5, step_size:float=0.03):
        """
        Summary:
        -------------
        Function to find a path for the agent to pick the specified object using KOMO optimization.
        
        Inputs:
        -------------
        - C: The configuration of the environment.
        - agent: The agent frame in the configuration.
        - obj: The object frame to be picked.
        - FS: List to store the frames used in the path.
        - verbose: Verbosity level for logging.
        - K: Number of iterations to try KOMO optimization.
        - N: Number of iterations to try RRT traj. generation.
        - step_size: Step size for the RRT pathfinding.
        
        Outputs:
        -------------
        - Returns True if a feasible path is found, otherwise returns False.
        -------------
        """

        if verbose > 0:
            print(f"Running Pick Path")

        Ct = ry.Config()
        Ct.addConfigurationCopy(C)
        base_col = abs(Ct.getCollisionsTotalPenetration())

        n = N
        max_iter = 1500
        for k in range(0, K):
            if verbose > 0:
                print(f"Trying Pick KOMO for {k}")

            # Make the robot go close to the object and than there find a feasible pick configuration
            if k == K-1:
                Ct.frame(obj).setContact(0)
                fr, p = self.run_rrt(Ct, Ct.frame(obj).getPosition()[:2], verbose, N=1, FS=[], step_size=step_size, max_iter=max_iter, delta_iter=2000, isOpt=True)
                Ct.frame(obj).setContact(1)
                if fr:
                    ag_sz = max(Ct.frame(agent).getSize())
                    ob_sz = max(Ct.frame(obj).getSize())
                    t_sz = (ag_sz + ob_sz)
                    bk = math.ceil(t_sz / step_size)+2
                    Ct.setJointState(p[max(0, len(p)-bk) ])
                else:
                    return False
                
            komo = ry.KOMO(Ct, phases=1, slicesPerPhase=1, kOrder=0, enableCollisions=True)  
            komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)                                                                                        # Randomize the initial configuration
            komo.addObjective([1], ry.FS.distance, [agent, obj], ry.OT.eq)      

            if k >= 1 and k < K-1:
                # Initialize robot from a random joint state (limited by joint limits)
                komo.initRandom()

            if k >= 3:
                n = 2

            ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve() 

            if verbose > 1:
                komo.view_play(True, f"Pick Komo Solution: {ret.feasible}, {ret.eq < 1}")
                komo.view_close()
                
            if ret.eq-base_col < 0.1 or ret.feasible:
                fr, _ = self.run_rrt(C, komo.getPath()[-1], verbose, N=n, FS=FS, step_size=step_size, max_iter=max_iter, delta_iter=2000, isOpt=True)
                if fr:
                    return True
                
        return False
    
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def run_rrt(self, C:ry.Config, goal:list, verbose: int, FS:list = None, N:int=20, step_size:float=0.05, isOpt:bool=True, max_iter:int=1000, delta_iter:int=1500):
        """
        Summary:
        -------------
        Function to run the RRT pathfinding algorithm to find a feasible path from the current configuration to the goal configuration.
        
        Inputs:
        -------------
        - C: The configuration of the environment.
        - goal: The target joint state to reach.
        - verbose: Verbosity level for logging.
        - FS: List to store the frames used in the path.
        - N: Number of iterations to try RRT pathfinding.
        - step_size: Step size for the RRT pathfinding.
        - isOpt: Boolean indicating whether to optimize the path using KOMO.
        - max_iter: Maximum number of iterations for the RRT pathfinding.
        - delta_iter: Increment to increase the maximum iterations if no feasible path is found.
        
        Outputs:
        -------------
        - Returns a tuple (feasible, path) where:
            - feasible: Boolean indicating if a feasible path was found.    
            - path: List of joint states representing the path if feasible, otherwise None.
        -------------
        """
        
        Ct = ry.Config()
        Ct.addConfigurationCopy(C)

        if verbose > 1:
            js = Ct.getJointState()
            Ct.view(True, "RRT Init")
            Ct.setJointState(goal)
            Ct.view(True, "RRT Final")
            Ct.view_close()
            Ct.setJointState(js)

        ry.params_clear()
        ry.params_add({"rrt/stepsize": step_size})
        ry.params_add({"rrt/maxIters": max_iter})

        for i in range(0, N):
            if verbose > 1:
                print(f"Running RRT for {max_iter} iterations with step size {step_size}")

            with suppress_stdout():
                rrt = ry.PathFinder()      
                if i > 0:
                    rrt.setProblem(Ct, Ct.getJointState(), goal, 1e-3)
                else:
                    rrt.setProblem(Ct, Ct.getJointState(), goal, 1e-3)
                s = rrt.solve()

            if s.feasible:
                path = s.x 
                
                if isOpt:
                    komo = ry.KOMO(Ct, phases=len(s.x), slicesPerPhase=1, kOrder=2, enableCollisions=True)                           
                    komo.addControlObjective([], 1)
                    komo.addControlObjective([], 2)
                    komo.initWithWaypoints(s.x, 1, False)
                    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)  
                    komo.addObjective([len(s.x)], ry.FS.qItself, [], ry.OT.eq, target=goal)  
                    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve() 

                    # If the object is not at the goal position, return failure
                    Ct2 = ry.Config()
                    Ct2.addConfigurationCopy(Ct)
                    Ct2.setFrameState(komo.getPathFrames()[-1])
                    rp = Ct2.getJointState()
                    if np.linalg.norm(np.array(rp)-np.array(goal)) > 1e-2:
                        return False, None

                    path = komo.getPath()

                if verbose > 1:

                    Ct.view(True, "RRT Solution: " + str(s.feasible))
                    for p in path:
                        Ct.setJointState(p)
                        Ct.view(False, "RRT Solution")
                        time.sleep(0.005)
                    Ct.view_close()

                if FS != None:
                    FS.extend(komo.getPathFrames())

                return True, path
            
            else:
                max_iter += delta_iter
                
                ry.params_clear()
                ry.params_add({"rrt/stepsize": step_size})
                ry.params_add({"rrt/maxIters": max_iter})
                
        return False, None
    
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def gen_adaptive_manip_plan(self, C:ry.Config, P:list, agent:str, obj:str, verbose: int, cfc_lim:int=5, K:int=5, slicesPerPhase:int=30, step_div:int=2, min_step:int=1):
        """
        Summary:
        -------------
        Function which adaptively selects placement subgoals from the given path and tries to find a feasible manipulation plan using KOMO optimization.
        
        Inputs:
        -------------
        - C: The configuration of the environment.
        - P: List of joint states representing the path to follow.
        - agent: The agent frame in the configuration.
        - obj: The object frame to be manipulated.
        - verbose: Verbosity level for logging.
        - cfc_lim: Consecutive feasible count limit to adjust the step size.
        - K: Number of iterations to try KOMO optimization.
        - slicesPerPhase: Number of slices per phase for KOMO optimization.
        - step_div: Factor to divide the step size when adjusting it.
        - min_step: Minimum step size.
        
        Outputs:
        -------------
        - Returns a tuple (is_feasible, seq_info, is_failed, FS, komo, pp_seq) where:
            - is_feasible: Boolean indicating if a feasible manipulation plan was found.
            - seq_info: List containing information about the subgoal sequence.
            - is_failed: Boolean indicating if the manipulation plan generation failed earlier but still moved the object to some extent.
            - FS: List of frames used in the motion plan.
            - komo: The KOMO object used for optimization.
            - pp_seq: Number of pick-and-place sequences in the manipulation plan.
        -------------
        """
        if self.verbose > 0:
            print("Solving for path")

        Ct = ry.Config()
        Ct.addConfigurationCopy(C)

        max_step = len(P) - 1
        step = len(P)-1
        pi = step
        cfc = 0 # consequtive feas count
        pf = 0
        pp_seq = 0
        seq_info = [] # List to store sequence information to be used in subgoal refinement (step_no, contact_point)
        is_placed = False
        prev_contact = None
        FS = []
        z = C.frame(obj).getPosition()[2]
        base_col = abs(Ct.getCollisionsTotalPenetration())
        while not is_placed:

            wp = P[pi]
            feasible = False
            
            for k in range(0, K):

                if verbose > 1:
                    print(f"pi:{pi}, tot: {len(P)-1}")
                    print("Step:", step)
                    print(f"Trying Move KOMO for {k+1} time")

                Ct.addFrame("subgoal", "world", "shape: marker, size: [0.1]").setPosition([*wp, z])
                komo = ry.KOMO(Ct, phases=2, slicesPerPhase=slicesPerPhase, kOrder=2, enableCollisions=True)   
                komo.addControlObjective([], 1, 1e-1)
                komo.addControlObjective([], 2, 1e-1)
                komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)                                                                                        # Randomize the initial configuration
                komo.addObjective([0.5,-1], ry.FS.distance, [agent, obj], ry.OT.eq)                                   
                komo.addModeSwitch([1,2], ry.SY.stable, [agent, obj], True)   
                komo.addObjective([2] , ry.FS.positionDiff, [obj, "subgoal"], ry.OT.eq)  
                
                if k > 0:
                    # Initialize robot from a random joint state (limited by joint limits)
                    komo.initRandom()
            
                ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve() 

                Ct.delFrame("subgoal")

                Ct2 = ry.Config()
                Ct2.addConfigurationCopy(Ct)
                Ct2.setFrameState(komo.getPathFrames()[-1])

                fin_col = abs(Ct2.getCollisionsTotalPenetration())
                ign_col = abs(Ct2.eval(ry.FS.negDistance, [obj, agent])[0])

                feasible = ret.eq-base_col*slicesPerPhase*2 < 0.15 and fin_col - base_col - ign_col <= 1e-2

                if verbose > 2:
                    komo.report(True, True, True)
                    komo.view_play(True, f"Move Komo Solution: {ret.feasible}, {ret.eq}, {feasible}, {base_col*slicesPerPhase*2}")
                    komo.view_close()
                
                if feasible :
                    FS.extend(komo.getPathFrames())
                    pp_seq += 1
                    Ct.setFrameState(komo.getPathFrames()[-1])

                    cur_contact = Ct.eval(ry.FS.positionRel, [obj, agent])[0]
                    if prev_contact is None:
                        seq_info.append([wp, 1, komo.getPathFrames()])
                    else:
                        seq_info.append([wp, np.linalg.norm(np.array(prev_contact)-np.array(cur_contact)), komo.getPathFrames()])
                    prev_contact = [*cur_contact]

                    if pi == max_step:
                        is_placed = True
                        break

                    # If the object can be moved cfc times, increase the step size
                    if cfc >= cfc_lim:
                        step = min(max_step, step*step_div)

                    pf = pi
                    pi = min(pf + step, max_step)

                    if max_step - pi < max_step * 0.05 and step > 2:
                        pi = max_step

                    cfc += 1
                    break
                
            if not feasible: 
                cfc = 0
                # If the step size is too small, exit the generation
                if step == min_step:
                    return False, seq_info, True, FS, komo, pp_seq
                
                # Reduce the step size and try again
                step = int(step / step_div)
                step = max(min_step, step)
                pi = min(pf + step, len(P)-1)

        if verbose > 2:
            self.display_solution(FS, pause=0.01)

        return True, seq_info, False, FS, komo, pp_seq

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def refine_manip_plan(self, C:ry.Config, seq_info:list, agent:str, obj:str, slicesPerPhase:int=30, FS:list=[], pp_seq:int=0):
        """
        Summary:
        -------------
        Function to refine the selected subgoals during the object placement plan.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - seq_info: List containing information about the subgoal sequence.
        - agent: The agent frame in the configuration.
        - obj: The object frame to be manipulated.
        - slicesPerPhase: Number of slices per phase for KOMO optimization.
        - FS: List of frames used in the motion plan.
        - pp_seq: Number of pick-and-place sequences in the manipulation plan.

        Outputs:
        -------------
        - Returns the updated number of pick-and-place sequences in the manipulation plan. The motion plan is updated in place in the FS list.
        """

        Ct = ry.Config()
        Ct.addConfigurationCopy(C)

        prev_frames = None
        i = -1
        z = C.frame(obj).getPosition()[2]
        base_col = abs(Ct.getCollisionsTotalPenetration())
        while i < len(seq_info)-1:
            i+=1
            
            if prev_frames is None:
                prev_frames = [*seq_info[i][2]]
                FS.extend(prev_frames)
                continue
            
            # If the contact point change is small, try to merge the subgoals
            if seq_info[i][1] < 0.1:
                k = i
                for j in range(i+1, len(seq_info)):
                    if seq_info[j][1] > 0.1:
                        break 
                    k += 1

                Ct.setFrameState(prev_frames[-1])
                wp_i = seq_info[k][0]

                Ct.addFrame("subgoal", "world", "shape: marker, size: [0.1]").setPosition([*wp_i, z])
                komo = ry.KOMO(Ct, phases=2, slicesPerPhase=slicesPerPhase, kOrder=2, enableCollisions=True)   
                komo.addControlObjective([], 1, 1e-1)
                komo.addControlObjective([], 2, 1e-1)
                komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=1e2)                                                                                        # Randomize the initial configuration
                komo.addObjective([1], ry.FS.distance, [agent, obj], ry.OT.eq, scale=1e2, target=0)                                     # Pick
                komo.addModeSwitch([1,2], ry.SY.stable, [agent, obj], True)   
                komo.addObjective([2] , ry.FS.positionDiff, [obj, "subgoal"], ry.OT.eq, scale=1e0)  # Place constraints 
                ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve() 
                Ct.delFrame("subgoal")

                if ret.eq - base_col*slicesPerPhase*2 < 0.1:
                    pp_seq -= k - i
                    FS.extend(komo.getPathFrames())
                    prev_frames = [*komo.getPathFrames()]
                    i = k
                    
                else:
                    FS.extend(seq_info[i][2])
                    prev_frames = [*seq_info[i][2]]

            else:
                FS.extend(seq_info[i][2])
                prev_frames = [*seq_info[i][2]]
                
        return pp_seq
    
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def scene_adjust(self, img:np.ndarray):
        """
        Summary:
        -------------
        Adjust the scene colors for Global Occupancy Matrix (GOM) generation.

        Inputs:
        -------------
        - img: The input RGB image as a numpy array.

        Outputs:
        -------------
        - Returns the adjusted image as a numpy array.
        """

        # Extract RGB channels
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        # Create boolean masks for conditions
        red_dominant = (G * 1.2 < R) & (B * 1.2 < R) & (R > 100)
        green_dominant = (R * 1.2 < G) & (B * 1.2 < G) & (G > 100)
        blue_dominant = (R * 1.2 < B) & (G * 1.2 < B) & (B > 100)
        yellow_dominant = ((B * 1.2 < R) & (B * 1.2 < G) & (R > 100) & (G > 100))


        img[red_dominant] = [0, 0, 0]
        img[green_dominant] = [80, 80, 80]
        img[blue_dominant | (yellow_dominant & ~green_dominant) ] = [200, 200, 200]
        img = img[:, :, 0]
        return img

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def scene_score(self, C:ry.Config, cam_frame:str, verbose:int=0):
        """
        Summary:
        -------------
        Function to compute the scene score based on the Global Occupancy Matrix (GOM) and Reachability Matrix.
        
        Inputs:
        -------------
        - C: The configuration of the environment.
        - cam_frame: The frame of the camera to use for generating the scene score.
        - verbose: Verbosity level for logging.
        
        Outputs:
        -------------
        - Returns the computed scene score as a float value.
        -------------
        """

        ry.params_clear()
        ry.params_add({"Render/useShadow":False})
        
        camera_view = ry.CameraView(C)
        cam = C.getFrame(cam_frame)
        camera_view.setCamera(cam)
        img, _ = camera_view.computeImageAndDepth(C)
        img = np.asarray(img)
        img_c = copy.deepcopy(img)

        # Extract RGB channels
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        
        # Create boolean masks for conditions
        red_dominant = (G * 1.2 < R) & (B * 1.2 < R) & (R > 50)
        green_dominant = (R * 1.2 < G) & (B * 1.2 < G) & (G > 50)
        blue_dominant = (R * 1.2 < B) & (G * 1.2 < B) & (B > 50)
        yellow_dominant = ((B * 1.2 < R) & (B * 1.2 < G) & (R > 50) & (G > 50))

        img[red_dominant] = [0, 0, 0]
        img[green_dominant] = [80, 80, 80]
        img[blue_dominant | (yellow_dominant & ~green_dominant) ] = [200, 200, 200]
        img = img[:, :, 0]

        img_c[~yellow_dominant] = [0, 0, 0]
        img_c[green_dominant] = [0, 0, 0]

        # The position of the agent in the scene
        try:
            coords_mat = np.argwhere(img_c)[0][:2]
        except:
            coords_mat = [150, 150]
        
        # Starting from the agent position, we will expand the reachability area using wavefront propagation
        h, w = img.shape[:2]
        dist = np.full((h, w), 0.99, dtype=float)

        # Generate the reachability matrix
        ego_h, ego_w = int(self.ego_size[0]), int(self.ego_size[1])  # (height, width)
        rh = ego_h // 2 -3
        rw = ego_w // 2 -3

        sy, sx = coords_mat
        q = deque()
        visited = np.zeros((h, w), dtype=bool)

        def block_in_bounds(y, x):
            return (y - rh >= 0) and (y + rh < h) and (x - rw >= 0) and (x + rw < w)

        if block_in_bounds(sy, sx):
            dist[sy - rh: sy + rh + 1, sx - rw: sx + rw + 1] = 1.1
            q.append((sy, sx))
            visited[sy, sx] = True  # mark when enqueued
            
        # 4-connected moves (step of 3)
        moves = [(rh//2, 0), (-rh//2, 0), (0, rw//2), (0, -rw//2), (rh//2, rw//2), (rh//2, -rw//2), (-rh//2, rw//2), (-rh//2, -rw//2)]

        while q:
            y, x = q.popleft()
            for dy, dx in moves:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and block_in_bounds(ny, nx):
                    blk = img[ny - rh: ny + rh + 1, nx - rw: nx + rw + 1]
                    
                    if (blk.flatten() > 0).all():
                        dist[ny - rh: ny + rh + 1, nx - rw: nx + rw + 1] = 1.1
                        q.append((ny, nx))
                        visited[ny, nx] = True

        # Introduce the reachability score to the scene score
        scene_score_raw = sum(img.flatten())

        weigted_img = img * dist
        weigted_img_n = weigted_img / (h*w*1e1)
        scene_score = sum(weigted_img_n.flatten())

        if verbose > 1:
            _, axes = plt.subplots(1, 3, figsize=(15, 6))
            imgs  = [img, dist, weigted_img]
            titles = ['Image', 'Reachability', 'Weighted Image, Score: ' + str(scene_score)]

            for i, (ax, im, title) in enumerate(zip(axes, imgs, titles)):
                ax.imshow(im, cmap='viridis' if i == 0 else "gray")
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            plt.show()  
        
        for o in self.obj_g_list:
            C.frame("goal" + o[3:]).setColor([0, 0, 1])

        return scene_score, scene_score_raw

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
    @staticmethod
    def cam_to_target(pts, cam_frame, target_frame):
        """
        Summary:
        -------------
        Function to transform points from the camera frame to the target frame.
        
        Inputs:
        -------------
        - pts: Numpy array of points in the camera frame.
        - cam_frame: The frame of the camera in the configuration.
        - target_frame: The frame of the target in the configuration.
        
        Outputs:
        -------------
        - Returns a numpy array of points transformed to the target frame.
        -------------
        """

        pts = pts[~np.all(pts == 0, axis=1)]  

        if pts.shape[0] == 0:
            return np.empty((0, 3)) 
        
        t_cam_world = cam_frame.getPosition()  
        R_cam_world = cam_frame.getRotationMatrix()  

        t_target_world = target_frame.getPosition() 
        R_target_world = target_frame.getRotationMatrix()  

        R_target_cam = np.dot(R_target_world, R_cam_world.T)  
        t_target_cam = t_target_world - np.dot(R_target_world, t_cam_world) 

        points_camera_frame_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1)))) 

        transformation_matrix = np.vstack((
            np.hstack((R_target_cam, t_target_cam.reshape(-1, 1))), 
            np.array([0, 0, 0, 1])
        ))  

        points_target_frame_homogeneous = np.dot(transformation_matrix, points_camera_frame_homogeneous.T).T

        points_target_frame = points_target_frame_homogeneous[:, :3]

        return points_target_frame
           
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
    def display_solution(self, FS:list=None, pause:float=0.02, isPathView:bool=False, isOnlyPath:bool=False):
        """
        Summary:
        -------------
        Function to display the solution by visualizing the frames in the configuration.
        
        Inputs:
        -------------
        - FS: List of frame states to visualize. If None, uses the current frame states.
        - pause: Pause duration between frame updates for visualization.
        - isPathView: Boolean indicating whether to visualize the path or just the final positions.

        Outputs:
        -------------
        - None
        -------------
        """
        ry.params_clear()
        ry.params_add({"Render/lights":[-5.,0.,4., -5.,0.,4.]})
        Ct = ry.Config()
        Ct.addConfigurationCopy(self.C)

        if FS == None:
            FS = self.FS

        if not isOnlyPath:
            Ct.view(True, "Solution")

        travel_dist = 0
        prev_pos = None
        for i, fs in enumerate(FS):
            Ct2 = ry.Config()
            Ct2.addConfigurationCopy(self.C)
            Ct2.setFrameState(fs)

            # If trajectory visualization is enabled, display the agent and object positions as silhouettes
            if isPathView:
                if i % 20 == 0:
                    sat = 0.5
                    agent_size = Ct2.frame(self.agent).getSize()
                    obj_size = Ct2.frame(self.obj).getSize()
                    Ct.addFrame(self.agent+"_"+str(i), "world", "shape:ssCylinder, size: [" + str(agent_size[0]) + " " + str(agent_size[1]) + " 0.1], color:[1 1 0 "+str(sat) +"]").setPosition(Ct2.frame(self.agent).getPosition())
                    Ct.addFrame(self.obj+"_"+str(i), "world", "shape:ssBox, size: [" + str(obj_size[0]) + " " + str(obj_size[1]) + " 0.15 0.02], color:[0 0 1 "+str(sat) +"]").setPosition(Ct2.frame(self.obj).getPosition())
                    for o in self.obs_list:
                        o_size = Ct2.frame(o).getSize()
                        Ct.addFrame(o+"_"+str(i), "world", "shape:ssBox, size: [" + str(o_size[0]) + " " + str(o_size[1]) + " 0.12 0.02], color:[1 1 1 "+str(sat) +"]").setPosition(Ct2.frame(o).getPosition())
                            
            Ct.frame(self.agent).setPosition(Ct2.frame(self.agent).getPosition())
            Ct.frame(self.obj).setPosition(Ct2.frame(self.obj).getPosition())
            for o in self.obs_list:
                Ct.frame(o).setPosition(Ct2.frame(o).getPosition())

            if prev_pos is not None:
                travel_dist += np.linalg.norm(np.array(Ct.frame(self.agent).getPosition()) - np.array(prev_pos))
            prev_pos = Ct.frame(self.agent).getPosition()

            if not isOnlyPath:
                Ct.view(False, "Solution")
                time.sleep(pause)

        if not isOnlyPath:
            Ct.view(True, "Solution")

        return travel_dist
    



