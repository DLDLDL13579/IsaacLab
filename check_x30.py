#!/usr/bin/env python3
"""
Script to check X30.usd file structure in Isaac Sim.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# import isaaclab modules
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Check X30.usd file structure.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage

# create a world
world = World(stage_units_in_meters=1.0)

# add X30.usd to the stage
x30_usd_path = "X30.usd"
prim_path = "/World/X30"
add_reference_to_stage(x30_usd_path, prim_path)

# get the X30 prim
x30_prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)

if x30_prim.IsValid():
    print(f"Successfully loaded X30.usd at: {prim_path}")
    
    # print prim hierarchy
    print("\nX30 Prim Hierarchy:")
    def print_hierarchy(prim, indent=0):
        indent_str = "  " * indent
        print(f"{indent_str}{prim.GetPath().name}: {prim.GetTypeName()}")
        for child in prim.GetChildren():
            print_hierarchy(child, indent + 1)
    
    print_hierarchy(x30_prim)
    
    # check for articulation
    from omni.isaac.core.articulations import Articulation
    try:
        articulation = Articulation(prim_path)
        print(f"\nX30 is an articulation with {articulation.num_joints} joints")
        print(f"Joint names: {articulation.joint_names}")
        print(f"DOF names: {articulation.dof_names}")
    except Exception as e:
        print(f"\nX30 is not an articulation: {e}")
else:
    print(f"Failed to load X30.usd at: {prim_path}")

# close the simulator
simulation_app.close()
