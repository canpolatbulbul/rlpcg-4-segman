import robotic as ry
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy 
import gurobipy as gp
from gurobipy import GRB
from MOSeGMan.SeGManv2 import SeGManv2
from collections import defaultdict
import os
from contextlib import contextmanager

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

class MOSeGMan():
    """
    Summary:
    -------------
    Multi-Objective Selective Guided Manipulation (MOSeGMan) class for robotic manipulation tasks.
    First it generates object placement sequence. Afterwards for each object in the placement order it iteratively solves the SeGMan problem.

    Inputs:
    -------------
    - C: The configuration of the environment.
    - C_hm: The auxilary configuration of the environment.
    - verbose: Verbosity level for logging.

    Outputs:
    -------------
    - The motion plan.

    """

    def __init__(self, C:ry.Config, C_hm:ry.Config, verbose:int):
        self.C = C
        self.C_hm = C_hm
        self.obj_g_list = []
        self.obj_m_list = []
        self.obs_list = []
        self.verbose = verbose
        self.tot_pp_seq = 0
        self.all_objs = []
        self.Chm_dict = defaultdict(ry.Config)
        self.dist_dict = defaultdict(lambda: defaultdict(float))
        self.dist_upd = defaultdict(lambda: defaultdict(int))
        self.col_map = defaultdict(set)
        self.col_diff = defaultdict(set)
        self.G = nx.DiGraph()
        self.io_diff = defaultdict(int)
        self.SINK = "end"


        frames = C.getFrameNames()
        for f in frames:
            attr = C.frame(f).getAttributes()
            atk = attr.keys()
            if "logical" in atk:
                atk_l = attr["logical"].keys()
                if "agent" in atk_l:
                    self.agent = f
                    self.START = f
                elif "movable_go" in atk_l:
                     self.obj_g_list.append(f)
                     self.all_objs.append(f)
                elif "movable_o" in atk_l:
                     self.obj_m_list.append(f)
                     self.all_objs.append(f)

        self.obs_list = copy.deepcopy(self.obj_m_list)
        self.segman = SeGManv2(self.C, self.C_hm, self.agent, self.obj_g_list, self.all_objs, self.obs_list, self.verbose)

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def run(self):
        """
        Summary:
        -------------
        Main function of the MO-SeGMan algorithm.
        It generates the order of objectives, and iteratively generates motion plan for the placement of each object.

        Outputs:
        -------------
        - Returns True if all objects are placed successfully.
        - Prints the total time taken for the entire process.
        -------------
        """

        tic_base = time.time()

        reloc_o = []
        seq_gen_time = 0

        # Generate the order of objectives based precedence relationship and further optimize it for the robot travel distance
        if len(self.obj_g_list) > 1:
            obj_g_order, gen_time = self.generate_object_placement_seq(self.C, self.obj_g_list, reloc_o, self.verbose)
            seq_gen_time += gen_time
        else:
            obj_g_order = self.obj_g_list

        # Initial verification of object placement
        placement_verified, placed_objs = self.verify_object_placement(obj_g_order)

        # Iteratively solve for each object in the order until all are placed correctly
        iter_lim = len(self.obj_g_list)+1
        iter = -1
        skip_c = 0
        skip_l = 3
        replan_c = 0
        while not all(placement_verified) and iter < iter_lim:

            iter += 1
            for i, obj in enumerate(obj_g_order):

                if not placement_verified[i]:
                    # Assume the goal objects are also possible obstacles
                    self.obs_list = copy.deepcopy(self.obj_m_list)
                    self.obs_list.extend(self.obj_g_list)
                    self.obs_list.remove(obj)

                    self.segman.obs_list = copy.deepcopy(self.obs_list)
                    
                    # Call SeGMan solver
                    f, pp_seq, reloc_o = self.segman.solve(obj, placed_objs)

                    replan_c += len(reloc_o) + 1
                    self.sync_C()
                    self.tot_pp_seq += pp_seq
                    if f:
                        skip_c = 0
                        # Verify the placement of the objects 
                        placement_verified, placed_objs = self.verify_object_placement(obj_g_order)
                        not_placed = [o for o in obj_g_order if o not in placed_objs]

                        # Regenerate the object placement sequence if any goal object is relocated
                        if len(reloc_o) > 0 and len(not_placed) > 1:
                            obj_g_order, gen_time = self.generate_object_placement_seq(self.C, not_placed, reloc_o, self.verbose)
                            seq_gen_time += gen_time
                            obj_g_order = placed_objs + obj_g_order
                            break
                        
                    else:
                        skip_c += 1
                        if skip_l == skip_c:
                            skip_c = 0
                            not_placed = [o for o in obj_g_order if o not in placed_objs]
                            # Update object placement sequence for the current position of the robot
                            if len(not_placed) > 1:
                                obj_g_order, gen_time = self.generate_object_placement_seq(self.C, not_placed, reloc_o,  self.verbose)
                                seq_gen_time += gen_time
                                obj_g_order = placed_objs + obj_g_order
                            break

        tac_base = time.time()
        print("Total Time: ", tac_base - tic_base)

        if not all(placement_verified):
            return False, tac_base - tic_base, 0, seq_gen_time, replan_c

        print("All objects placed successfully!")
        return True, tac_base - tic_base, self.tot_pp_seq, seq_gen_time, replan_c

# -------------------------------------------------------------------------------------------------------------- #
    
    def display_solution(self, FS:list=None, pause:float=0.02, isPathView:bool=False, isOnlyPath:bool=False):
        """
        Summary:
        -------------
        Display the generated motion plan

        Input:
        -------------
        - FS: The motion plan to be displayed.
        - pause: Pause duration between frames. 
        - isPathView: Boolean flag to indicate if the path view is to be displayed.
        - isOnlyPath: Boolean flag to indicate to get the travel distance of the robot without visualization. 

        Outputs:
        - Returns the travel distance of the robot if isOnlyPath is True.    
        """

        return self.segman.display_solution(FS, pause, isPathView, isOnlyPath)

# -------------------------------------------------------------------------------------------------------------- #
    
    def sync_C(self):
        """
        Summary:
        -------------
        Function to synchronize the current configuration with the high-level configuration.
        """
        del self.C
        self.C = ry.Config()
        self.C.addConfigurationCopy(self.segman.C)

        if self.C_hm is not None:
            del self.C_hm
            self.C_hm = ry.Config()
            self.C_hm.addConfigurationCopy(self.segman.C_hm)
        

# -------------------------------------------------------------------------------------------------------------- #
    
    def sync_segman_C(self):
        """
        Summary:
        -------------
        Function to synchronize the current configuration with the high-level configuration.
        """
        del self.segman.C
        self.segman.C = ry.Config()
        self.segman.C.addConfigurationCopy(self.C)

        if self.segman.C_hm is not None:
            del self.segman.C_hm
            self.segman.C_hm = ry.Config()
            self.segman.C_hm.addConfigurationCopy(self.C_hm)

        self.segman.obj_g_list = copy.deepcopy(self.obj_g_list)
        self.segman.all_objs = copy.deepcopy(self.all_objs)

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def generate_object_placement_seq(self, C: ry.Config, obj_g_list: list, reloc_o: list, verbose:int = 0):
        """
        Summary:
        -------------
        Function to generate the order of placement for the goal objects by minimizing the distance traveled by the agent and amount of relocation required.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - obj_g_list: List of goal objects to be placed.
        - reloc_o: List of objects that have been relocated.
        - verbose: Verbosity level for logging.

        Outputs:
        -------------
        - Returns a list of objects in the order they should be placed and the generation time.
        -------------
        """
        tic_gen = time.time()
        n = len(obj_g_list)
        
        # The initial relaxed cost matrix (euclidiean distance) between the agent and the objects, and between the objects and their goals
        self.generate_dist_mat(C, obj_g_list, reloc_o)

        # Generate the trajectory for each object to their goal and check for collisions
        self.check_trajectory_collisions(C, obj_g_list, verbose)

        # Obtain the Directed Acyclic Graph (DAG) from the object dependencies and further checking and eliminating cycles
        Q, P = self.generate_DAG(obj_g_list, verbose)

        lim = 2
        iter = 0
        prev_z_best = []    
        z_best = []
        while iter < lim:
            # Run the Gurobi solver to find the optimal sequence of objects to be placed with constraints
            try:
                z_best = self.solve_sequence_gurobi(obj_g_list, n, P, Q)
            except Exception as e:
                print("Error occurred while solving sequence:", e)
                toc_gen = time.time()
                return list(z_best), toc_gen - tic_gen
            
            # If sequence is not changed with the updated cost matrix, break the loop
            if z_best == prev_z_best:
                break

            # Update the cost matrix with the RRT path lengths for the current sequence
            self.calc_seq_cost(C, z_best, obj_g_list, reloc_o)

            iter += 1
            prev_z_best = list(z_best)

        toc_gen = time.time()

        if verbose > 0:
            print("Optimal sequence found in:", toc_gen - tic_gen)
            print("Sequence:", z_best)

        return list(z_best), toc_gen - tic_gen

# -------------------------------------------------------------------------------------------------------------- #  

    def solve_sequence_gurobi(self, obj_g_list:list, n:int, P=None, Q=None):
        """
        Summary:
        -------------
        Function to solve the sequence of objects to be placed using Gurobi optimization.
        
        Inputs:
        -------------
        - obj_g_list: List of goal objects to be placed.
        - n: The number of objects.
        - P: Dictionary of parents for each object.
        - Q: Dictionary of children for each object.

        Outputs:
        -------------
        - Returns the optimal order of objects to be placed.
        -------------
        """

        arcs = []
        for fr in [self.agent, *obj_g_list]:
            for to in [*obj_g_list,self.SINK]:  
                if fr != to and not (fr == self.agent and to == self.SINK):
                    arcs.append((fr, to))
                    
        m = gp.Model("sequence_path")
        # suppress Gurobi output
        m.setParam('OutputFlag', 0)
        x = m.addVars(arcs, vtype=GRB.BINARY, name="x")

        # MTZ order vars
        u = {}
        u[self.START] = m.addVar(lb=0, ub=0, vtype=GRB.INTEGER, name="u_start")
        for o in obj_g_list:
            u[o] = m.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name=f"u_{o}")
        u[self.SINK] = m.addVar(lb=n+1, ub=n+1, vtype=GRB.INTEGER, name="u_sink")

        # Objective
        m.setObjective(gp.quicksum(self.dist_dict[a[0]][a[1]] * x[a] for a in arcs), GRB.MINIMIZE)

        # Degrees / path
        m.addConstr(gp.quicksum(x[self.START, o] for o in obj_g_list) == 1, "start_out")
        for o in obj_g_list:
            m.addConstr(gp.quicksum(x[o2, o] for o2 in [self.START] + [o2 for o2 in obj_g_list if o2 != o]) == 1, f"in_{o}")
        for o in obj_g_list:
            m.addConstr(gp.quicksum(x[o, o2] for o2 in [o2 for o2 in obj_g_list if o2 != o]) + x[o, self.SINK] == 1, f"out_{o}")
        m.addConstr(gp.quicksum(x[o, self.SINK] for o in obj_g_list) == 1, "into_sink")

        # MTZ (path)
        M = n + 1
        for o in [self.START] + obj_g_list:
            for o2 in obj_g_list:
                if o == o2: continue
                m.addConstr(u[o] + 1 <= u[o2] + M*(1 - x[o, o2]), name=f"mtz_{o}_{o2}")
        for o in obj_g_list:
            m.addConstr(u[o] + 1 <= u[self.SINK] + M*(1 - x[o, self.SINK]), name=f"mtz_sink_{o}")

        # Precedences from P (parents) and Q (children)
        added_prec = set()
        if P:
            for j_after in obj_g_list:
                for p_before in P.get(j_after, []):
                    if p_before == j_after or p_before not in obj_g_list: continue
                    pair = (p_before, j_after)
                    if pair in added_prec: continue
                    m.addConstr(u[p_before] + 1 <= u[j_after], name=f"prec_{p_before}_before_{j_after}")
                    added_prec.add(pair)
        if Q:
            for i_before in obj_g_list:
                for c_after in Q.get(i_before, []):
                    if c_after == i_before or c_after not in obj_g_list: continue
                    pair = (i_before, c_after)
                    if pair in added_prec: continue
                    m.addConstr(u[i_before] + 1 <= u[c_after], name=f"prec_{i_before}_before_{c_after}")
                    added_prec.add(pair)

        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(f"Gurobi ended with status {m.Status}")

        # Extract order
        succ = {}
        for (i, j) in arcs:
            if x[i, j].X > 0.5:
                succ[i] = j

        order = []
        cur = self.START
        seen = set()
        while cur in succ:
            nxt = succ[cur]
            if nxt == self.SINK or nxt in seen: break
            order.append(nxt)
            seen.add(nxt)
            cur = nxt

        return order

# -------------------------------------------------------------------------------------------------------------- #  

    def calc_seq_cost(self, C:ry.Config, seq:list, obj_g_list: list, reloc_o: list):
        """
        Summary:
        -------------
        Calculate the cost of a given sequence of objects based on RRT paths.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - seq: The sequence of objects to calculate the cost for.
        - dist_mat: The distance matrix between the objects and their goals.

        Outputs:
        -------------
        - Updates the distance matrix with the RRT path lengths for the sequence.
        -------------
        """
        
        Ct = ry.Config()
        Ct.addConfigurationCopy(C)
        for o in obj_g_list:
            Ct.frame(o).setContact(0)

        pr_o = seq[0]
        for i, o in enumerate(seq):

            if i == 0 and (self.dist_upd[self.START][o] != 1 or o in reloc_o) :
                f, p = self.segman.run_rrt(Ct, Ct.frame(o).getPosition()[:2], self.verbose, N=2, step_size=0.03, isOpt=False, delta_iter=2000)
                if f:
                    self.dist_dict[self.START][o] = len(p)*0.03
                    self.dist_upd[self.START][o] = 1
                else:
                    self.dist_dict[self.START][o] = 1e6   
                    self.dist_upd[self.START][o] = 0

            elif (self.dist_upd[pr_o][o] != 1 or o in reloc_o or pr_o in reloc_o):
                Ct.setJointState(Ct.frame("goal" + pr_o[3:]).getPosition()[:2])
                f, p = self.segman.run_rrt(Ct, Ct.frame(o).getPosition()[:2], self.verbose, N=2, step_size=0.03, isOpt=False, delta_iter=2000)
                if f:
                    self.dist_dict[pr_o][o] = len(p)*0.03 
                    self.dist_upd[pr_o][o] = 1
                else:
                    self.dist_dict[pr_o][o] = 1e6   
                    self.dist_upd[pr_o][o] = 0

            pr_o = o

# -------------------------------------------------------------------------------------------------------------- #

    def generate_dist_mat(self, C: ry.Config, obj_g_list: list, reloc_o: list):
        """
        Summary:
        -------------
        Function to generate the distance matrix between the agent, objects, and their goals.
        The distance matrix is constructed such that:
        - The first row corresponds to the agent's distance to each object.
        - The subsequent rows correspond to the distance from each object to its goal position, with the diagonal entries set to infinity.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - obj_g_list: List of goal objects to be placed.
        - reloc_o: List of objects that have been relocated.

        Outputs:
        -------------
        - Returns a distance matrix of shape (n+1, n) where:
            - The first row contains distances from the agent to each object.
            - The subsequent rows contain distances from each object to its goal position.
            - The diagonal entries are set to infinity to avoid self-loops.
        -------------
        """
        ego_pos  =  C.frame(self.agent).getPosition()[:2]
        obj_pos  = {obj: C.frame(obj).getPosition()[:2] for obj in obj_g_list}
        goal_pos = {obj: C.frame("goal" + str(obj[3:])).getPosition()[:2] for obj in obj_g_list}

        if len(reloc_o) == 0:
            for so in [self.agent, *obj_g_list]:
                for to in [*obj_g_list, self.SINK]:
                    if so == to:
                        continue
                    elif to == self.SINK:
                        self.dist_dict[so][to] = 0
                    elif so == self.agent:
                        self.dist_dict[so][to] = np.linalg.norm(np.array(ego_pos) - np.array(obj_pos[to]))
                    else:
                        self.dist_dict[so][to] = np.linalg.norm(np.array(ego_pos) - np.array(goal_pos[to]))
        else:
            for so in [self.agent, *obj_g_list]:
                for to in obj_g_list:
                    if so == to or (so not in reloc_o and to not in reloc_o):
                        continue
                    elif so == self.agent and to in reloc_o:
                        self.dist_dict[so][to] = np.linalg.norm(np.array(ego_pos) - np.array(obj_pos[to]))
                    else:
                        self.dist_dict[so][to] = np.linalg.norm(np.array(ego_pos) - np.array(goal_pos[to]))


# -------------------------------------------------------------------------------------------------------------- #

    def check_collisions(self, C: ry.Config, obj: str, obj_g_list: list, verbose: int = 0):
        """
        Summary:
        -------------
        Function to parse collisions in the configuration and update the collision map and collision difference dict.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - obj: The object for which collisions are being checked.
        - obj_g_list: List of goal objects to be placed.
        - verbose: Verbosity level for logging.

        Outputs:
        -------------
        - Updates the collision map and collision difference dict for the specified object.
        """

        def parse(name: str):
            if name.startswith('tmp'):  return 'tmp',  int(name[3:])
            if name.startswith('obj'):  return 'obj',  name[3:]
            if name.startswith('goal'): return 'goal', name[4:]
            return None, None

        col_pairs = set()
        with suppress_stdout():
            for o1, o2, col in C.getCollisions(0):
                
                if abs(col) < 1e-3: continue

                k1, i1 = parse(o1)
                k2, i2 = parse(o2)

                # --- Type 1: traj(o_i) X start(obj_j)
                if (k1 == 'tmp' and k2 == 'obj' and f'obj{i2}' in obj_g_list) or \
                (k2 == 'tmp' and k1 == 'obj' and f'obj{i1}' in obj_g_list):
                    j = i2 if k2 == 'obj' else i1
                    # Only add Type 1 if a Type 2 for the same j does NOT already exist
                    if (2, "obj"+j) not in col_pairs and "obj"+j in obj_g_list:
                        col_pairs.add((1, "obj"+j))


                # --- Type 2: traj(o_i) X goal(obj_j)
                elif (k1 == 'tmp' and k2 == 'goal' and f'obj{i2}' in obj_g_list) or \
                    (k2 == 'tmp' and k1 == 'goal' and f'obj{i1}' in obj_g_list):
                    j = i2 if k2 == 'goal' else i1
                    if "obj"+j in obj_g_list:
                        col_pairs.add((2, "obj"+j))

                    col_pairs.discard((1, "obj"+j))
            
            diff_add = {(*dia, 1) for dia in (col_pairs - self.col_map[obj])}
            diff_rem = {(*dir, -1) for dir in (self.col_map[obj] - col_pairs)}

            self.col_diff[obj] = copy.deepcopy(diff_add | diff_rem)
            self.col_map[obj] = copy.deepcopy(col_pairs)

            if verbose > 0:
                print("col_pairs:", sorted(col_pairs))
# -------------------------------------------------------------------------------------------------------------- #

    def check_trajectory_collisions(self, C: ry.Config, obj_g_list: list, verbose:int = 0):
        """
        Summary:
        -------------
        Function to check for trajectory collisions of objects in the environment.

        Inputs:
        -------------
        - C: The configuration of the environment.
        - obj_g_list: List of goal objects to be placed.
        - verbose: Verbosity level for logging.

        Outputs:
        -------------
        - Updates the collision map and collision difference dict for all objects.
        """

        Ct = ry.Config()
        Ct.addConfigurationCopy(C)
        
        for obj in obj_g_list:

            Ct2 = self.segman.make_agent(Ct, obj, self.agent)

            # Make obstacles contact free
            for o in self.obj_m_list:
                Ct2.frame(o).setContact(0)

            # Set all other objects to contact free except the current one
            for o in self.obj_g_list:
                if o != obj and o in obj_g_list:
                    Ct2.frame(o).setContact(0)
                else:
                    Ct2.frame(o).setContact(1)

            # Generate trajectory
            f, p = self.segman.run_rrt(Ct2, Ct2.frame("goal" + obj[3:]).getPosition()[:2], verbose, N=3, step_size=0.03, delta_iter=2500, isOpt=False)

            # Add trajectory frames to the configuration for collision checking
            if f:
                shape = Ct2.frame(obj).getShapeType()
                size  = Ct2.frame(obj).getSize()
                quat = Ct2.frame(obj).getQuaternion()
                z = Ct2.frame(obj).getPosition()[2]
                color = C.frame(obj).getAttributes()["color"]
                for i in range(0, len(p), 5):
                    if len(p) - i < 5:
                        pi = p[-1]
                    else:
                        pi = p[i]
                    clone = Ct2.addFrame("tmp" + str(i))
                    clone.setColor([color[0]*0.5,color[1]*0.5,color[2]*0.5 , 0.5])
                    clone.setShape(shape, size)
                    clone.setQuaternion(quat)
                    clone.setPosition([*pi, z*0.8])
                    clone.setContact(1)
                self.Chm_dict[obj] = Ct2
            else:
                continue


            for o in self.obj_g_list:
                if o in obj_g_list and obj != o:
                    Ct2.frame(o).setContact(1)
                    Ct2.frame("goal" + o[3:]).setContact(1)
                else:
                    Ct2.frame(o).setContact(0)
                    Ct2.frame("goal" + o[3:]).setContact(0)

            # Check for collisions with the generated trajectory and other objects
            self.check_collisions(Ct2, obj, obj_g_list, verbose)
            
            if verbose > 1:
                print(obj, self.col_map[obj])
                Ct2.view(True)
                Ct2.view_close()

        if verbose > 0:
            print(obj_g_list)
            print("Coll Diff", self.col_diff)

# -------------------------------------------------------------------------------------------------------------- #

    def generate_DAG(self, obj_g_list:list, verbose:int=0):
        """
        Summary:
        -------------
        Function to generate a Directed Acyclic Graph (DAG) from the collision map.
        The function creates a directed graph where nodes represent objects and edges represent dependencies based on collisions.
        
        Inputs:
        -------------
        - obj_g_list: List of goal objects to be placed.
        - verbose: Verbosity level for logging.

        Outputs:
        -------------
        - Returns the directed graph (DAG), children, and parents of each node.
        -------------
        """

        for obj in self.obj_g_list:
            if obj not in obj_g_list:
                if obj in self.G.nodes:
                    self.G.remove_node(obj)
                continue

            collisions = self.col_diff[obj]

            self.G.add_node(obj)
            if len(collisions) == 0:
                continue

            for (typ, obj2, a_r) in collisions:
                    if typ == 1:
                        if a_r == 1:
                            if not self.G.has_edge(obj2, obj):
                                self.G.add_edge(obj2, obj, weight=1)
                                self.io_diff[obj2] += 1
                                self.io_diff[obj] -= 1
                            else:
                                self.col_map[obj].discard((1, obj2))
                        else:
                            if self.G.has_edge(obj2, obj):
                                self.G.remove_edge(obj2, obj)
                                self.io_diff[obj2] -= 1
                                self.io_diff[obj] += 1
                    
                    else:
                        if a_r == 1:
                            if self.G.has_edge(obj, obj2):
                                self.G.remove_edge(obj, obj2)
                                self.io_diff[obj] -= 1
                                self.io_diff[obj2] += 1
                                self.col_map[obj2].discard((1, obj))

                            self.G.add_edge(obj, obj2, weight=2)
                            self.io_diff[obj] += 1
                            self.io_diff[obj2] -= 1
                        else:
                            if self.G.has_edge(obj, obj2):
                                self.G.remove_edge(obj, obj2)
                                self.io_diff[obj] -= 1
                                self.io_diff[obj2] += 1

        # Convert the graph to a Directed Acyclic Graph (DAG) by removing cycles
        G_orig = copy.deepcopy(self.G)
        self.convert_to_DAG(len(obj_g_list))

        # Obtain the children and parents
        children = dict()
        parents = dict()
        for node in self.G.nodes:
            children[node] = set()
            parents[node] = set() 
            for des in nx.descendants(self.G, node):
                children[node].add(des)
            for anc in nx.ancestors(self.G, node):
                parents[node].add(anc)
    
        if verbose > 1:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            pos = nx.spring_layout(G_orig)

            self.draw_graph(pos, G_orig, axes[0], "Before DAG (with cycles)")
            self.draw_graph(pos, self.G, axes[1], "After DAG (acyclic)")
            plt.tight_layout()
            plt.show()

        return children, parents

# -------------------------------------------------------------------------------------------------------------- #
    def convert_to_DAG(self, n):
        """
        Summary:
        -------------
        Function to convert a directed graph to a Directed Acyclic Graph (DAG) by removing cycles using Depth-First Search (DFS) based method.

        Inputs:
        -------------
        - n: The number of nodes in the graph.

        Outputs:
        -------------
        - Returns the total number of edges removed to convert the graph to a DAG.
        """

        tic = time.time()
        nodes = list(self.G.nodes())
        stack = []
        track = []
        track2rem = {}
        cyc_id = 0
        tot_rem = 0

        def dfs_search(node: any, G: nx.DiGraph):
            nonlocal cyc_id, tot_rem

            if node in stack:
                # Greedy removal of edges in the cycle if number of objects are more than 20
                if n > 20:
                    rem_edge = track[-1]
                    if self.G.has_edge(rem_edge[0], rem_edge[1]):
                        if self.G[rem_edge[0]][rem_edge[1]]['weight'] == 1:
                            self.col_map[rem_edge[1]].discard((1, rem_edge[0]))
                        else:
                            self.col_map[rem_edge[0]].discard((2, rem_edge[1]))
                        self.G.remove_edge(rem_edge[0], rem_edge[1])
                        tot_rem += 1
                    return
                cyc_id += 1
                idx = -1

                # Add edges to the cycle dict with the same cycle id
                while idx >= -len(track):
                    track2rem.setdefault(tuple(track[idx]), []).append(cyc_id)
                    if track[idx][0] == node:  # closing edge reached
                        break
                    idx -= 1
                return

            stack.append(node)
            for child in list(G.successors(node)):
                track.append((node, child))
                dfs_search(child, G)
                track.pop()
            stack.pop()

            if node in nodes:
                nodes.remove(node)

        while nodes:
            stack.clear()
            track.clear()
            dfs_search(nodes[0], self.G)

        # Remove the edges
        while track2rem and n <= 20:
            max_edge = None
            max_ids_len = 0
            max_ids = 0
            min_w = float('inf')

            # Remove the edge with maximum cycle ids and minimum weight
            for edge, rem_ids in list(track2rem.items()):
                if len(rem_ids) > max_ids_len or len(rem_ids) == max_ids_len and self.G[edge[0]][edge[1]]['weight'] < min_w or \
                    len(rem_ids) == max_ids_len and self.G[edge[0]][edge[1]]['weight'] == min_w and self.io_diff[max_edge[0]] > self.io_diff[edge[0]]:
                    max_edge = edge
                    max_ids_len = len(rem_ids)
                    max_ids = rem_ids
                    min_w = self.G[edge[0]][edge[1]]['weight']

            del track2rem[max_edge]
            self.G.remove_edge(max_edge[0], max_edge[1])
            tot_rem += 1
            if min_w == 1:
                self.col_map[max_edge[1]].discard((1, max_edge[0]))
            else:
                self.col_map[max_edge[0]].discard((2, max_edge[1]))

            max_ids_set = set(max_ids)
            for e, ids in list(track2rem.items()):
                n_ids = [x for x in ids if x not in max_ids_set]
                if len(n_ids) == 0:
                    del track2rem[e]
                    continue
                track2rem[e] = n_ids
        toc = time.time()
        print(f"Time taken to convert to DAG: {toc - tic:.4f} seconds")
        return tot_rem

# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------- #

    def verify_object_placement(self, obj_g_order:list, epsilon:float=0.1):
        """
        Summary:
        -------------
        Function to verify if the objects in the given order are placed correctly in the environment.
        
        Inputs:
        -------------
        - obj_g_order: List of objects in the order to be verified.
        - epsilon: Tolerance for the position verification.

        Outputs:
        -------------
        - Returns a boolean array indicating whether each object is placed correctly.
        -------------
        """

        verified = np.zeros(len(obj_g_order), dtype=bool)
        placed_objs = []
        for i, obj in enumerate(obj_g_order):
            if np.linalg.norm(self.C.frame(obj).getPosition()[:2] - self.C.frame("goal" + obj[3:]).getPosition()[:2] ) < epsilon:
                verified[i] = True
                placed_objs.append(obj)
            else:
                verified[i] = False

        if self.verbose > 0:
            print("Object placement order: ", obj_g_order)
            print("Object placement verification: ", verified)

        return verified, placed_objs

# -------------------------------------------------------------------------------------------------------------- #

    def draw_graph(self, pos, Gx, ax, title):
        nx.draw_networkx_nodes(Gx, pos, node_color='lightblue', node_size=1000, ax=ax)
        nx.draw_networkx_labels(Gx, pos, ax=ax)
        strong_edges = [(u, v) for u, v, d in Gx.edges(data=True) if d['weight'] == 2]
        weak_edges = [(u, v) for u, v, d in Gx.edges(data=True) if d['weight'] == 1]
        nx.draw_networkx_edges(Gx, pos, edgelist=strong_edges, style='solid', width=2, ax=ax,
                            arrows=True, arrowstyle='->', min_source_margin=15, min_target_margin=15)
        nx.draw_networkx_edges(Gx, pos, edgelist=weak_edges, style='dashed', width=2, ax=ax,
                            connectionstyle="arc3,rad=0.2", arrows=True,
                            arrowstyle='->', min_source_margin=15, min_target_margin=15)
        ax.set_title(title)
        ax.axis('off')