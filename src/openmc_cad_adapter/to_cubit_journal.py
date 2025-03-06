from argparse import ArgumentParser
from collections.abc import Iterable
import math
from numbers import Real
from pathlib import Path
import sys
import warnings

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
from .cubit_util import *

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
    reset_cubit_ids()
    global cell_ids, cmds, off_center

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
    cell_map = {}
    uni_map = {}
    latt_map = {}
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
    
    def midp(node): # Returns the midpoint of a bounding box
        return ' '.join( map(str, (node.bounding_box.upper_right+node.bounding_box.lower_left)/2) )
    
    def process_bb(bbox, w): # Returns array of x,y,z lengths of a bounding box object
        w2 = np.abs(bbox.upper_right-bbox.lower_left)
        w_out = []
        for i in range(3):
            if np.isfinite(w2[i]):
                w_out += [w2[i]]
            else:
                w_out += [w[i]]
        return w_out

    def trim_uni(node, ids, w):
        #TODO fix move in surfaces, YCyl,.., remove cad_cmds
        w = process_bb(node.bounding_box, w)
        exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
        s = body_id()
        added = False
        for id in ids: 
            inter_ids = np.append(np.array(id), np.array(s))
            s1 = body_id()
            exec_cubit( f"intersect volume {to_cubit_list(inter_ids)} keep" )
            s_inter = body_id()
            if last_id(s1) + 1 != last_id(s_inter) and s_inter != s1: # If multiple volumes are created they are saves as a multivolume body
                    exec_cubit( f"split body {to_cubit_list(mul_body_id)}" ) # Split the multivolume body
            s2 = body_id() # Resulting intersection ids
            if not added: # Not all intersections return a volume 
                if s2 != s: # catch the id of first created one to return
                    added = True
                    strt = first_id(s2)
            if s1 != s2: # Link material to new volume
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
        trim_ids = range(strt, stp + 1, 1)
        return trim_ids
    
    def trim_cell_like(ids, s_ids):
        s = body_id()
        added = False
        for id in ids:
            inter_ids = np.append(np.array(id), np.array(s_ids))
            s1 = body_id()
            exec_cubit( f"intersect volume {to_cubit_list(inter_ids)} keep" )
            s_inter = body_id()
            if last_id(s1) + 1 != last_id(s_inter) and s_inter != s1: # If multiple volumes are created they are saves as a multivolume body
                    exec_cubit( f"split body {to_cubit_list(mul_body_id)}" ) # Split the multivolume body
            s2 = body_id() # Resulting intersection ids
            if not added: # Not all intersections return a volume 
                if s2 != s: # catch the id of first created one to return
                    added = True
                    strt = first_id(s2)
            if s1 != s2: # Link material to new volume
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
        trim_ids = range(strt, stp + 1, 1)
        return trim_ids
        
    def surface_to_cubit_journal(node, w, hex = False):
        global surf_coms, cell_ids, off_center
        if isinstance(node, Halfspace):
            try:
                surface = node.surface
            except: 
                surface = node._surface
            if cad_surface := _CAD_SURFACE_DICTIONARY.get(surface._type):
                cad_surface = cad_surface.from_openmc_surface(surface)
                return cad_surface.to_cubit_surface(type(node),node, w, False, hex, off_center)
            else:
                raise NotImplementedError(f"{surface.type} not implemented")
        elif isinstance(node, Complement):
            id = surface_to_cubit_journal(node.node, w)
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            wid = body_id()
            exec_cubit( f"subtract volume {{ {id} }} from volume {{ {wid} }} keep_tool" )
            return np.array(body_id())
        elif isinstance(node, Intersection):
            #TODO only one volume should be returned
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            inter_id = body_id()
            for subnode in node:
                s = surface_to_cubit_journal( subnode, w )
                max_id = np.max(np.append(inter_id,s))
                exec_cubit( f"intersect volume {' '.join( map(str, np.append(np.array(inter_id),np.array(s))) )} keep" )
                if max_id + 1 != last_id(body_id()): # If multiple volumes are created they are saves as a multivolume body
                    exec_cubit( f"split body {to_cubit_list(mul_body_id())}" ) # Split the multivolume body
                inter_id = body_id()
            return np.array(inter_id).astype(int)
        elif isinstance(node, Union):
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            union_id = body_id()
            first = surface_to_cubit_journal( node[0], w )
            max_id = np.max(np.append(union_id, first))
            exec_cubit( f"intersect volume {' '.join( map(str, np.append(np.array(union_id),np.array(first))) )} keep" )
            if max_id + 1 != last_id(body_id()): # If multiple volumes are created they are saves as a multivolume body
                exec_cubit( f"split body {to_cubit_list(mul_body_id)}" ) # Split the multivolume body
            union_id = body_id()
            for subnode in node[1:]:
                s = surface_to_cubit_journal( subnode, w )
                exec_cubit( f"unite volume {' '.join( map(str, np.append(np.array(union_id),np.array(s))) )} keep" )
                union_id = body_id()
            return np.array(union_id).astype(int)
        else:
            raise NotImplementedError(f"{node} not implemented")

    def process_node( node, bb ):
        # TODO propagate names, check if bb is centred in 0,0,0 or moved, FIXME in z0 move by half world
        # TODO fix geom_util and others
        global surf_coms, cell_ids
        
        # Universes contain cells and move internal cells to proper location
        if isinstance( node, Universe ): 
            if node.id not in uni_map:
                ids = np.array([])
                for c in node._cells.values():
                    ids = np.append(ids,np.array(process_node( c, bb))).astype(int)
                #exec_cubit( f'create group "uni_{node.id}"' )
                uni_map[node.id] = ids
            ids = uni_map[node.id]
            exec_cubit(f"brick x {world[0]} y {world[1]} z {world[2]}\n")
            strt = body_id() + 1
            exec_cubit( f" volume { to_cubit_list(ids) } copy" )
            stp = body_id()
            ids3 = range(strt,stp+1,1)
            for a in range(len(ids3)):
                try:
                    cell_mat[ids3[a]] = cell_mat[ids[a]]
                except:
                    pass
            exec_cubit( f"move volume {to_cubit_list(ids3)} midpoint location {midp(node)}" )
            ids_out = trim_uni(node, ids3, bb)
            return ids_out
        
        elif isinstance( node, Cell ): # Cell instance that is moved to proper location by universe
            if node.id not in cell_map:
                ids = np.array([])
                #TODO handle single cell conversions
                if isinstance( node.fill, Material ):
                    s_ids = surface_to_cubit_journal(node.region, bb)
                    ids = np.append(ids,np.array(s_ids)).astype(int)
                    for id in ids:
                        cell_mat[id] = node.fill.name
                    
                elif node.fill is None:
                    s_ids = surface_to_cubit_journal(node.region, bb)
                    ids = np.append(ids,np.array(s_ids)).astype(int)
                    for id in ids:
                        cell_mat[id] = "void"
                
                elif isinstance( node.fill, Iterable ):
                    s_ids = surface_to_cubit_journal(node.region, bb)
                    ids2 = []
                    for uni in node.fill:
                        ids2 = np.append(ids2, np.array(process_node( uni, bb ))).astype(int)
                    ids = np.append(ids,np.array(trim_cell_like(ids2, s_ids))).astype(int)
                    
                else:
                    s_ids = surface_to_cubit_journal(node.region, bb)
                    ids2 = np.array(process_node( node.fill, bb )).astype(int)
                    ids = np.append(ids,np.array(trim_cell_like(ids2, s_ids))).astype(int)

                if isinstance( node.fill, Material ) or node.fill is None:
                    if node.name is None:
                        #exec_cubit( f'create group "cell_{node.id}"' )
                        pass
                    else:
                        exec_cubit( f'Volume {to_cubit_list(ids)}  rename "cell_{node.name}"' )
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
                    i = 0
                    for row in node.universes:
                        j = 0
                        for u in row:
                            for cell in u._cells.values():
                                #TODO check if proper order i,j or j,i
                                x = j * dx
                                y = i * dy
                                ids2 = process_node( cell, bb )
                                exec_cubit(f"brick x {world[0]} y {world[1]} z {world[2]}\n")
                                strt = body_id() + 1
                                exec_cubit( f" volume {to_cubit_list(ids2)} copy" )
                                stp = body_id()
                                ids3 = range(strt,stp+1,1)
                                for a in range(len(ids3)):
                                    try:
                                        cell_mat[ids3[a]] = cell_mat[ids2[a]]
                                    except:
                                        pass
                                exec_cubit( f"move volume {to_cubit_list(ids3)} midpoint location {x} {y} 0" )
                                ids = np.append(ids, np.array(ids3).astype(int)).astype(int)
                            j = j + 1
                        i = i + 1
                else:
                    raise NotImplementedError(f"{node} not implemented")
                latt_map[node.id] = ids
            return latt_map[node.id]
            
            
    
        # #FIXME rotate and tranlate
        # r = flatten( results )
        # if len( r ) > 0:
        #     if node.name:
        #         exec_cubit( f"body {{ {r[0]} }} name \"{node.name}\"" )
        #     else:
        #         exec_cubit( f"body {{ {r[0]} }} name \"Cell_{node.id}\"" )
        # return r
    
    def propagate_mat(id):
        if isinstance(cell_mat[id], str):
            return cell_mat[id]
        else:
            return propagate_mat(cell_mat[id])
      
    def process_mat(mat_n, id):
        if mat_n not in mat_map:
            exec_cubit( f'create material name "{mat_n}" ' )
            b_id = block_next()
            exec_cubit( f'Block {b_id} add volume {id}' )
            exec_cubit( f'Block {b_id} material "{mat_n}"' )
            mat_map[mat_n] = b_id
        else:
            exec_cubit( f'Block {mat_map[mat_n]} add volume {id}' )
        
    
    # Initialize commands
    # exec_cubit("set echo off\n")
    # exec_cubit("set info off\n")
    # exec_cubit("set warning off\n")
    exec_cubit("graphics pause\n")
    # #exec_cubit("set journal off\n")
    # exec_cubit("set default autosize off\n")
    exec_cubit("undo off\n")
    
    # Initialize world
    exec_cubit(f"brick x {world[0]} y {world[1]} z {world[2]}\n")
    
    # Process geometry
    
    if all(np.array(midp(geom.root_universe).split(" ")).astype(float) == 0):
        off_center = 0
    else:
        off_center = 1
    final_ids = process_node(geom.root_universe, w)
    
    # Process materials
    for id in final_ids:
        mat_n = propagate_mat(id)
        process_mat(mat_n, id)
    
    # Cleanup
    for i in range(1,last_id(final_ids)+1,1):
        if i not in final_ids:
            #if i in [16,59,327,414,415,416,417,443,487,491]: continue
            exec_cubit( f"delete volume {{ {i} }}" )
    
    #Finalize
    exec_cubit("graphics flush\n")
    exec_cubit("set default autosize on\n")
    exec_cubit("zoom reset\n")
    exec_cubit("set echo on\n")
    exec_cubit("set info on\n")
    exec_cubit("set warning on\n")
    exec_cubit("set journal on\n")


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