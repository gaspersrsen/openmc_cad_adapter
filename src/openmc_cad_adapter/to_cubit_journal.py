from argparse import ArgumentParser
from collections.abc import Iterable
import math
from numbers import Real
from pathlib import Path
import sys
import warnings
import copy

import numpy as np

try:
    import openmc
except ImportError as e:
    raise type(e)("Please install OpenMC's Python API to use the CAD conversion tool")

from openmc.region import Region, Complement, Intersection, Union
from openmc.surface import Halfspace, Quadric
from openmc.lattice import Lattice, HexLattice, RectLattice
from openmc import Universe, Cell, Material
from openmc import Plane, XPlane, YPlane, ZPlane, XCylinder, YCylinder, ZCylinder, Sphere, Cone
from openmc import XCone, YCone, ZCone, XTorus, YTorus, ZTorus

from .gqs import *

try:
    sys.path.append('/opt/Coreform-Cubit-2025.1/bin/')
    from .conv_cubit_API import *
except ImportError:
    raise ImportError("Cubit Python API not found. Please install Cubit to use this feature.")

from .surfaces import _CAD_SURFACE_DICTIONARY, surf_map

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def to_cubit_journal(geometry : openmc.Geometry,
                     materials : openmc.Materials,
                     world : Iterable[Real] = None,
                     cells: Iterable[int, openmc.Cell] = None,
                     filename: str = "openmc.jou",
                     to_cubit: bool = True):
    """Convert an OpenMC geometry to a Cubit journal.

    Parameters
    ----------
        geometry : openmc.Geometry
            The geometry to convert to a Cubit journal.
        world : Iterable[Real], optional
            Extents of the model in X, Y, and Z. Defaults to None.
        cells : Iterable[int, openmc.Cell], optional
            List of cells or cell IDs to write to individual journal files. If None,
            all cells will be written to the same journal file. Defaults to None.
        filename : str, optional
            Output filename. Defaults to "openmc.jou".
        to_cubit : bool, optional
            Uses the cubit Python module to write the model as a .cub5 file.
            Defaults to False.

    """
    global center_world

    if not filename.endswith('.jou'):
        filename += '.jou'

    if isinstance(geometry, openmc.Model):
        geometry = geometry.geometry

    if cells is not None:
        cells_ids = [c if not isinstance(c, openmc.Cell) else c.id for c in cells]
    else:
        cell_ids = []


    geom = geometry

    if world is None:
        bbox = geometry.bounding_box
        world = np.abs(bbox.upper_right-bbox.lower_left)
        if not all(np.isfinite(world)):
            raise RuntimeError('Model bounds were not provided and the bounding box determined by OpenMC is not finite.'
                               ' Please provide a world size argument to proceed')
    if world is None:
        raise RuntimeError("Model extents could not be determined automatically and must be provided manually")

    w = world
    bbox_world = geometry.bounding_box
    center_world = [0,0,0] # Init, changed later
    cell_map = {}
    uni_map = {}
    latt_map = {}
    inter_map = {}
    mat_map = {}
    cell_mat = {}
    
    def to_cubit_list(ids):
        if type(ids) != int:
            ids = ' '.join( map(str, np.array(ids) ))
        return ids
    
    def first_id(ids): # Returns the first id of input ids
        out = None
        try:
            for a in ids:
                out = a
                break
        except:
            out = ids
        return out
    
    def last_id(ids): # Returns the last id of input ids
        out = None
        try:
            for a in ids:
                out = a
        except:
            out = ids
        return out
    
    def midp(bb): # Returns the midpoint of a bounding box
        global center_world
        mid_dist = np.array(bb.upper_right)/2 + np.array(bb.lower_left)/2
        if all(np.isfinite(mid_dist)):
            return mid_dist
        else:
            return center_world
        
    def process_bb(bbox, w): # Returns array of x,y,z lengths of a bounding box object
        w2 = np.abs(bbox.upper_right-bbox.lower_left)
        w_out = []
        for i in range(3):
            if np.isfinite(w2[i]):
                w_out += [w2[i]]
            else:
                w_out += [w[i]]
        return w_out

    def trim_uni(node, ids, w): #TODO too time consuming
        w = process_bb(node.bounding_box, w)
        exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
        s = volume_id()
        added = False
        strt=-1
        for id in ids: 
            inter_ids = np.append(np.array(id), np.array(s))
            s1 = volume_id()
            exec_cubit( f"intersect volume {to_cubit_list(inter_ids)} keep" )
            s_inter = volume_id()
            if last_id(s1) + 1 != last_id(s_inter) and s_inter != s1: # If multiple volumes are created they are saves as a multivolume body
                    exec_cubit( f"split body {to_cubit_list(mul_body_id())}" ) # Split the multivolume body
            s2 = volume_id() # Resulting intersection ids
            if not added: # Not all intersections return a volume 
                if s2 != s: # catch the id of first created one to return
                    added = True
                    strt = s_inter + 1
            if s1 != s2: # Link materials to new volumes
                try:
                    for a in range(len(s2)):
                        try:
                            cell_mat[s2[a]] = cell_mat[id]
                        except:
                            raise ValueError(f"Volume {id} has no material")
                except:
                    try:
                        cell_mat[s2] = cell_mat[id]
                    except:
                        raise ValueError(f"Volume {id} has no material")
        if strt ==-1:
            return []
        stp = last_id(s2)
        trim_ids = range(strt, stp + 1, 1)
        return trim_ids
    
    def trim_cell_like(ids, s_ids): #TODO this is too time consuming
        if len(ids) == 0:
            raise ValueError(f"Ids {ids} is empty") 
        s = volume_id()
        added = False
        for id in ids:
            inter_ids = np.append(np.array(id), np.array(s_ids))
            s1 = volume_id()
            exec_cubit( f"intersect volume {to_cubit_list(inter_ids)} keep" )
            s_inter = volume_id()
            if last_id(s1) + 1 != last_id(s_inter) and s_inter != s1: # If multiple volumes are created they are saves as a multivolume body
                    exec_cubit( f"split body {to_cubit_list(mul_body_id())}" ) # Split the multivolume body
            s2 = volume_id() # Resulting intersection ids
            print(s2)

            if s1 != s2: # Link materials to new volumes
                if not added: # Not all intersections return a volume, catch the id of first created one to return
                    added = True
                    strt = np.min([first_id(s2),first_id(s_inter)]) #TODO WIERD
                try:
                    for a in range(len(s2)):
                        try:
                            cell_mat[s2[a]] = cell_mat[id]
                        except:
                            raise ValueError(f"Volume {id} has no material")
                except:
                    try:
                        cell_mat[s2] = cell_mat[id]
                    except:
                        raise ValueError(f"Volume {id} has no material")           
        stp = last_id(s2)
        try:
            trim_ids = range(strt, stp + 1, 1)
            return trim_ids
        except:
            warnings.warn(f"All cells have been trimmed:\n cells {ids} \n surfaces {s_ids}")
            return np.array([])
        
    def surface_to_cubit_journal(node, w, bb, hex = False):
        global surf_coms, cell_ids, center_world
        print(node)
        if isinstance(node, Halfspace):
            try:
                surface = node.surface
            except: 
                surface = node._surface
            if cad_surface := _CAD_SURFACE_DICTIONARY.get(surface._type):
                cad_surface = cad_surface.from_openmc_surface(surface)
                return cad_surface.to_cubit_surface(type(node), node, w, inner_world=None, hex=hex, off_center=center_world)
            else:
                raise NotImplementedError(f"{surface.type} not implemented")
        elif isinstance(node, Complement):
            #TODO ADD DICT
            id = surface_to_cubit_journal(node.node, w, bb)
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            wid = volume_id()
            exec_cubit( f"subtract volume {{ {id} }} from volume {{ {wid} }} keep_tool" )
            return np.array(volume_id())
        elif isinstance(node, Intersection):
            if str(node) not in inter_map:
                exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                inter_id = np.array(volume_id()).astype(int)
                #TODO not needed can itersect first node with second, check null intersection, else return null
                strt = volume_id() + 1
                for subnode in node:
                    s = surface_to_cubit_journal( subnode, w, bb )
                    strt_in = volume_id() + 1
                    # if type(s) != int:
                    #     raise ValueError(f"surface id {s} is not int")
                    if inter_id.size > 1:
                        raise NotImplementedError(f"{node, subnode} intersection split")
                        next_ids = np.array([])
                        for id in inter_id:
                            max_id = np.max(np.append(np.append(inter_id,s),next_ids))
                            strt = int(max_id + 1)
                            exec_cubit( f"intersect volume {' '.join( map(str, np.append(np.array(id),np.array(s))) )} keep" )
                            if max_id + 1 != last_id(volume_id()): # If multiple volumes are created they are saves as a multivolume body
                                exec_cubit( f"split body {to_cubit_list(mul_body_id())}" ) # Split the multivolume body
                            stp = last_id(volume_id())
                            next_ids = np.append(next_ids,np.array(range(strt,stp+1,1))).astype(int)
                    else:
                        max_id = np.max(np.append(inter_id,s))
                        strt = int(max_id + 1)
                        exec_cubit( f"intersect volume {' '.join( map(str, np.append(np.array(inter_id),np.array(s))) )} keep" )
                        if strt_in == last_id(volume_id()) + 1:
                            continue
                        elif strt_in != last_id(volume_id()): # If multiple volumes are created they are saves as a multivolume body
                            exec_cubit( f"split body {to_cubit_list(mul_body_id())}" ) # Split the multivolume body
                        stp = last_id(volume_id())
                        next_ids = np.array(range(strt,stp+1,1))
                    inter_id = np.array(next_ids).astype(int)
                inter_map[str(node)] = np.array(inter_id).astype(int)
            return inter_map[str(node)]
        elif isinstance(node, Union):
            out = np.array([])
            for subnode in node:
                s = surface_to_cubit_journal( subnode, w, bb )
                out = np.append( out, np.array(surface_to_cubit_journal( subnode, w, bb )) )
            strt= volume_id() + 1
            exec_cubit( f"unite volume {to_cubit_list(out)} keep" )
            end = volume_id()
            return np.array(end).astype(int)
            #return np.array(out).astype(int)
        else:
            raise NotImplementedError(f"{node} not implemented")

    def process_node( node, w, bb ):
        # TODO propagate names, check if bb is centred in 0,0,0 or moved
        global surf_coms, cell_ids, center_world
        
        # Universes contain cells and move internal cells to proper location
        if isinstance( node, Universe ): 
            if node.id not in uni_map:
                ids = np.array([])
                for c in node._cells.values():
                    ids = np.append(ids,np.array(process_node( c, w, midp(node.bounding_box)-center_world))).astype(int)
                uni_map[node.id] = ids
            ids = uni_map[node.id]
            exec_cubit(f"brick x {world[0]} y {world[1]} z {world[2]}\n")
            strt = volume_id() + 1
            exec_cubit( f" volume { to_cubit_list(ids) } copy" )
            stp = last_id(volume_id())
            ids3 = range(strt,stp+1,1)
            for a in range(len(ids3)):
                try:
                    cell_mat[ids3[a]] = cell_mat[ids[a]]
                except:
                    pass
            move_vec = center_world-midp(node.bounding_box)
            if any(move_vec != 0):
                exec_cubit( f"volume {to_cubit_list(ids3)} move {to_cubit_list(move_vec)}" )
            return ids3
            # ids_out = trim_uni(node, ids3, bb)
            # return ids_out
        
        elif isinstance( node, Cell ): # Cell instance that is moved to proper location by universe
            if node.id not in cell_map:
                ids = np.array([])
                if isinstance( node.fill, Material ):
                    s_ids = surface_to_cubit_journal(node.region, w, bb)
                    ids = np.append(ids,np.array(s_ids)).astype(int)
                    for id in ids:
                        cell_mat[int(id)] = node.fill.name
                    
                elif node.fill is None:
                    s_ids = surface_to_cubit_journal(node.region, w, bb)
                    ids = np.append(ids,np.array(s_ids)).astype(int)
                    for id in ids:
                        cell_mat[id] = "void"
                
                elif isinstance( node.fill, Iterable ):
                    s_ids = surface_to_cubit_journal(node.region, w, bb)
                    ids2 = []
                    for uni in node.fill:
                        ids2 = np.append(ids2, np.array(process_node( uni, w, bb ))).astype(int)
                    ids = np.append(ids,np.array(trim_cell_like(ids2, s_ids))).astype(int)
                    
                else:
                    ids2 = np.array(process_node( node.fill, w, bb )).astype(int)
                    if ids2.size != 0:
                        s_ids = surface_to_cubit_journal(node.region, w, bb)
                        ids = np.append(ids,np.array(trim_cell_like(ids2, s_ids))).astype(int)

                # if isinstance( node.fill, Material ) or node.fill is None:
                #     if node.name is None:
                #         pass
                #     else:
                #         exec_cubit( f'Volume {to_cubit_list(ids)}  rename "cell_{node.name}"' )
                cell_map[node.id] = ids
            return cell_map[node.id]
                
        elif isinstance( node, RectLattice ):
            if node.id not in latt_map: #General lattice that has to be copied and moved
                ids = np.array([])
                if node.ndim == 2:
                    ids = []
                    pitch = node._pitch
                    dx = pitch[0]
                    dy = pitch[1]
                    [nx, ny] = node.shape
                    x0 = -(nx-1)/2 * dx
                    y0 = -(ny-1)/2 * dy
                    
                    exec_cubit(f"brick x {dx} y {dy} z {w[2]}")
                    base_rect = volume_id()
                    i = 0
                    for row in node.universes:
                        j = 0
                        for u in row:
                            for cell in u._cells.values():
                                #TODO check if proper order i,j or j,i
                                #TODO chceck proper movement, is it center or lower left
                                x = j * dx
                                y = i * dy
                                ids2 = process_node( cell, w, bb )#midp(node.bounding_box) )
                                if ids2.size == 0:
                                    continue
                                
                                strt = last_id(volume_id()) + 1
                                exec_cubit( f" volume {to_cubit_list(ids2)} copy" )
                                stp = last_id(volume_id())
                                ids3 = list(range(strt,stp+1,1))
                                for a in range(len(ids3)):
                                    cell_mat[ids3[a]] = cell_mat[ids2[a]]
                                ids4 = trim_cell_like(ids3, base_rect)
                                exec_cubit( f"volume {to_cubit_list(ids4)} move {x+x0} {y+y0} 0" )
                                ids = np.append(ids, np.array(ids4)).astype(int)
                            j = j + 1
                        i = i + 1
                else:
                    raise NotImplementedError(f"{node} not implemented")
                # ADD OUTER WORLD
                exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                wid = volume_id()
                #strt = wid+1
                exec_cubit( f"subtract volume {to_cubit_list(ids)} from volume {wid} keep_tool" )
                #stp = last_id(volume_id())
                ids5 = np.array(volume_id())#range(strt,stp+1,1)
                for a in range(len(ids5)): #TODO fix outer, which can be other than material
                    outer_cell_mat = [cell.fill.name for cell in node.outer._cells.values()]
                    cell_mat[ids5[a]] = outer_cell_mat[0]
                ids = np.append(ids, np.array(ids5)).astype(int)
                latt_map[node.id] = ids
            return np.array(latt_map[node.id])
    
    def propagate_mat(id):
        if isinstance(cell_mat[id], str):
            return cell_mat[id]
        else:
            cell_mat[id] = propagate_mat(cell_mat[id])
            return cell_mat[id]
        
    def process_materials(ids):
        vals = cell_mat.values()
        u_vals = []
        for val in vals:
            if val not in u_vals:
                u_vals += [val]
        for mat in u_vals:
            mat_ids = [k for (k,v) in cell_mat.items() if v is mat and k in ids]
            exec_cubit( f'create material name "{mat}" ' )
            b_id = block_next()
            exec_cubit( f'Block {b_id} add volume {to_cubit_list(mat_ids)}' )
            exec_cubit( f'Block {b_id} material "{mat}"' )
            
    # Initialize commands
    # exec_cubit("set echo off\n")
    # exec_cubit("set info off\n")
    # exec_cubit("set warning off\n")
    exec_cubit("graphics pause\n")
    exec_cubit("undo off\n")
    
    # Initialize world
    exec_cubit(f"brick x {2*world[0]} y {2*world[1]} z {2*world[2]}\n")
    
    # Process geometry
    center_world = midp(geom.root_universe.bounding_box)
    final_ids = process_node(geom.root_universe, w, center_world)
    
    # Process materials
    # for id in final_ids:
    #     propagate_mat(id)
    #     #process_mat(mat_n, id)
    # process_materials(final_ids)
    
    # Cleanup
    del_ids = np.array([])
    for i in range(1,np.max(final_ids)+1,1):
        if i not in final_ids:
            del_ids = np.append(del_ids, i)
    exec_cubit( f"delete volume  {to_cubit_list(del_ids.astype(int))} " )
    
    #Finalize
    exec_cubit("graphics flush\n")
    exec_cubit("zoom reset\n")
    exec_cubit("set echo on\n")
    exec_cubit("set info on\n")
    exec_cubit("set warning on\n")
    exec_cubit("set journal on\n")
    exec_cubit("Save cub5 'cubit_model.cub5' Overwrite")


def openmc_to_cad():
    """Command-line interface for OpenMC to CAD model conversion"""
    parser = ArgumentParser()
    parser.add_argument('input', help='Path to a OpenMC model.xml file or path to a directory containing OpenMC XMLs')
    parser.add_argument('-o', '--output', help='Output filename', default='openmc.jou')
    parser.add_argument('-w', '--world-size', help='Maximum width of the geometry in X, Y, and Z', nargs=3, type=int)
    parser.add_argument('-c', '--cells', help='List of cell IDs to convert', nargs='+', type=int)
    parser.add_argument('--to-cubit', help='Run  Cubit', default=False, action='store_true')
    parser.add_argument('--cubit-path', help='Path to Cubit bin directory', default=None, type=str)
    args = parser.parse_args()

    model_path = Path(args.input)

    if model_path.is_dir():
        if not (model_path / 'settings.xml').exists():
            raise IOError(f'Unable to locate settings.xml in {model_path}')
        model = openmc.Model.from_xml(*[model_path / f for f in ('geometry.xml', 'materials.xml', 'settings.xml')])
    else:
        model = openmc.Model.from_model_xml(model_path)

    if args.cubit_path is not None:
        sys.path.append(args.cubit_path)

    to_cubit_journal(model.geometry, model.materials, world=args.world_size, filename=args.output, cells=args.cells, to_cubit=args.to_cubit)


__all__ = ['CADPlane', 'CADXPlane', 'CADYPlane', 'CADZPlane',
           'CADCylinder', 'CADXCylinder', 'CADYCylinder', 'CADZCylinder',
           'CADSphere', 'CADXCone', 'CADYCone', 'CADZCone', 'CADXTorus', 'CADYTorus', 'CADZTorus']