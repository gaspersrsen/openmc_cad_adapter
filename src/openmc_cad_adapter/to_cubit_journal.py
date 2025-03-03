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

def to_cubit_journal(geometry : openmc.Geometry, world : Iterable[Real] = None,
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
    global cell_ids, cmds

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
    
    def midp(node):
        return ' '.join( map(str, (node.bounding_box.upper_right+node.bounding_box.lower_left)/2) )
    
    def process_bb(bbox, w):
        w2 = np.abs(bbox.upper_right-bbox.lower_left)
        w_out = []
        for i in range(3):
            if np.isfinite(w2[i]):
                w_out += [w2[i]]
            else:
                w_out += [w[i]]
        return w_out

    def trim_uni(node, ids, w):
        #TODO what if a whole cell is cut-off and _CUBIT_ID is not created
        #TODO fix move in surfaces, YCyl,.., remove cad_cmds
        w = process_bb(node.bounding_box, w)
        exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
        s = body_id()
        #s = surface_to_cubit_journal( node.region, w)
        strt = body_id() +1
        #all_ids = np.append(np.array(ids), np.array(s))
        #print(f"{{ {' '.join( map(str, np.array(all_ids)) )} }}")
        
        #exec_cubit( f"intersect volume {{ {ids} }} {{ {s} }} keep" )
        added = False
        for id in ids:
            inter_ids = np.append(np.array(id), np.array(s))
            exec_cubit( f"intersect volume {' '.join( map(str, np.array(inter_ids)) )} keep" )
            if not added:
                if body_id() != s:
                    added = True
                    strt = body_id()
            exec_cubit( f"delete volume {{ {id} }}" )
        exec_cubit( f"delete volume {{ {s} }}" )
        stp = body_id()
        if strt > stp:
            raise ValueError(f"Universe {node} trim unsuccessful")
        trim_ids = range(strt, stp, 1)
        return trim_ids
        

    def surface_to_cubit_journal(node, w, hex = False):
        global surf_coms, cell_ids
        if isinstance(node, Halfspace):
            try:
                surface = node.surface
            except: 
                surface = node._surface
            if cad_surface := _CAD_SURFACE_DICTIONARY.get(surface._type):
                cad_surface = cad_surface.from_openmc_surface(surface)
                return cad_surface.to_cubit_surface(type(node),node, w, hex)
            else:
                raise NotImplementedError(f"{surface.type} not implemented")
        elif isinstance(node, Complement):
            id = surface_to_cubit_journal(node.node, w)
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            wid = body_id()
            exec_cubit( f"subtract volume {{ {id} }} from volume {{ {wid} }} keep_tool" )
            return wid
        elif isinstance(node, Intersection):
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            strt = body_id() + 1
            inter_id = body_id()
            for subnode in node:
                s = surface_to_cubit_journal( subnode, w)
                exec_cubit( f"intersect volume {{ {inter_id} }} {{ {s} }} keep" )
                exec_cubit( f"delete volume {{ {inter_id} }}" )
                inter_id = body_id()
            return np.array(range(strt, inter_id,1)).astype(int)
        elif isinstance(node, Union):
            exec_cubit( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            union_id = body_id()
            first = surface_to_cubit_journal( node[0], w,  + 1, )
            exec_cubit( f"intersect volume {{ {union_id} }} {{ {first} }} keep" )
            exec_cubit( f"delete volume {{ {union_id} }}" )
            union_id = body_id()
            for subnode in node[1:]:
                s = surface_to_cubit_journal( subnode, w,  + 1, )
                exec_cubit( f"unite volume {{ {union_id} }} {{ {s} }} keep" )
                exec_cubit( f"delete volume {{ {union_id} }}" )
                union_id = body_id()
            exec_cubit( f"delete volume {{ {s} }}" )
            return union_id
        else:
            raise NotImplementedError(f"{node} not implemented")


    def process_node( node, bb, surfs=None, lat_pos=None ):
        # TODO handle uni move and region, copied volume has to be trimmed
        global surf_coms, cell_ids
        
        # Universes contain cells and move internal cells to proper location
        if isinstance( node, Universe ): 
            if node.id not in uni_map:
                ids = np.array([])
                for c in node._cells.values():
                    #ids = np.append(ids,np.array(process_node( c, process_bb(node.bounding_box, w) ))).astype(int)
                    ids = np.append(ids,np.array(process_node( c, bb))).astype(int)
                #exec_cubit( f'create group "uni_{node.id}"' )
                uni_map[node.id] = ids
            ids = uni_map[node.id]
            strt = body_id() +1
            exec_cubit( f" volume {{ {' '.join( map(str, np.array(ids)) )} }} copy" )
            stp = body_id()
            ids3 = range(strt,stp,1)
            exec_cubit( f"move volume {strt} to {stp} midpoint location {midp(node)}" )
            trim_uni(node, ids3, bb)
            return ids3
        
        elif isinstance( node, Cell ): # Cell instance that is moved to proper location by universe
            if node.id not in cell_map:
                ids = np.array([])
                #TODO add bb, handle single cell conversions
                if isinstance( node.fill, Material ):
                    #s_ids = surface_to_cubit_journal(node.region, process_bb(node.bounding_box, w))
                    s_ids = surface_to_cubit_journal(node.region, bb)
                    mat_identifier = f"mat:{node.fill.id}"
                    # use material names when possible
                    if node.fill.name is not None and node.fill.name:
                        mat_identifier = f"mat:{node.fill.name}"
                    if len(mat_identifier) > 32:
                        mat_identifier = mat_identifier[:32]
                        warnings.warn(f'Truncating material name {mat_identifier} to 32 characters')
                    #exec_cubit( f'group \"{mat_identifier}\" add volume {{ { s_ids } }} ' )
                    #print(s_ids,ids)
                    ids = np.append(ids,np.array(s_ids)).astype(int)
                
                elif node.fill is None:
                    #s_ids = surface_to_cubit_journal(node.region, process_bb(node.bounding_box, w))
                    s_ids = surface_to_cubit_journal(node.region, bb)
                    #exec_cubit( f'group "mat:void" add volume {{ { s_ids } }} ' )
                    ids = np.append(ids,np.array(s_ids)).astype(int)
                
                elif isinstance( node.fill, Iterable ):
                    for uni in node.fill:
                        #ids = np.append(ids, np.array(process_node( uni, process_bb(node.bounding_box, w) ))).astype(int)
                        ids = np.append(ids, np.array(process_node( uni, bb ))).astype(int)
                
                else:
                    #ids = np.append(ids, np.array(process_node( node.fill, process_bb(node.bounding_box, w) ))).astype(int)
                    ids = np.append(ids, np.array(process_node( node.fill, bb ))).astype(int)

                # if node.id in cell_ids:
                #     write_journal_file(f"{filename[:-4]}{node.id}.jou", surf_coms[start:], process_bb(node.bounding_box, w))
                if node.name is None:
                    #exec_cubit( f'create group "cell_{node.id}"' )
                    pass
                    #exec_cubit( f'create group "cell_{node.id}"' )
                else:
                    exec_cubit( f'body {{ {ids} }} Rename "cell_{node.name}"' )
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
                                id = process_node( cell, [ dx, dy, w[2] ])
                                ids2 = str( id )
                                if isinstance( id, list ):
                                    ids2 = ' '.join( map(str, id) )
                                strt = body_id() +1
                                exec_cubit( f" volume {{ {ids2} }} copy" )
                                stp = body_id()
                                ids3 = range(strt,stp,1)
                                exec_cubit( f"move volume {strt} to {stp} midpoint location {x} {y} 0" )
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
    
    # Initialize world
    # exec_cubit("set echo off\n")
    # exec_cubit("set info off\n")
    # exec_cubit("set warning off\n")
    # exec_cubit("graphics pause\n")
    # #exec_cubit("set journal off\n")
    # exec_cubit("set default autosize off\n")
    # exec_cubit("undo off\n")
    exec_cubit(f"brick x {world[0]} y {world[1]} z {world[2]}\n")
    # Process geometry
    # final_ids = []
    # for cell in geom.root_universe._cells.values():
    #     final_ids += process_node(cell, w)
    final_ids = process_node(geom.root_universe, w)
    
    # Cleanup
    for i in range(1,body_id() +1,1):
        if i not in final_ids:
            exec_cubit( f"delete volume {{ {i} }}" )
        # found = False
        # for j in [surf_map]:
        #     if i in j.values:
        #         exec_cubit( f"delete volume {{ {i} }}" )
        # #for j in [cell_map,uni_map,latt_map]:
        #     if i in j.values:
        #         found = True
        #         break
        # if not found:
        #     exec_cubit( f"delete volume {{ {i} }}" )
    #Finalize
    exec_cubit("graphics flush\n")
    exec_cubit("set default autosize on\n")
    exec_cubit("zoom reset\n")
    exec_cubit("set echo on\n")
    exec_cubit("set info on\n")
    exec_cubit("set warning on\n")
    exec_cubit("set journal on\n")


def write_journal_file(filename, surf_coms, world, verbose_journal=False):
    with open(filename, "w") as f:
        if not verbose_journal:
            f.write("set echo off\n")
            f.write("set info off\n")
            f.write("set warning off\n")
            f.write("graphics pause\n")
            f.write("set journal off\n")
            f.write("set default autosize off\n")
            f.write("undo off\n")
            f.write(f"brick x {world[0]} y {world[1]} z {world[2]}\n")
            
        for x in surf_coms:
            f.write(x + "\n")
        if not verbose_journal:
            f.write("graphics flush\n")
            f.write("set default autosize on\n")
            f.write("zoom reset\n")
            f.write("set echo on\n")
            f.write("set info on\n")
            f.write("set warning on\n")
            f.write("set journal on\n")
    print(surf_map)


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

    to_cubit_journal(model.geometry, world=args.world_size, filename=args.output, cells=args.cells, to_cubit=args.to_cubit)


__all__ = ['CADPlane', 'CADXPlane', 'CADYPlane', 'CADZPlane',
           'CADCylinder', 'CADXCylinder', 'CADYCylinder', 'CADZCylinder',
           'CADSphere', 'CADXCone', 'CADYCone', 'CADZCone', 'CADXTorus', 'CADYTorus', 'CADZTorus']