from abc import ABC, abstractmethod
import sys
import math
import warnings

import numpy as np
import openmc

from .geom_util import move, rotate
from .conv_cubit_API import *


def indent(indent_size):
    return ' ' * (2*indent_size)


surf_map = {}

def surf_id(node):
    return node.surface.id * (-1 if node.side == "-" else 1)

class CADSurface(ABC):

    def to_cubit_surface(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        ids_map = self.to_cubit_surface_inner(ent_type, node, extents, inner_world, hex, off_center)
        # TODO: Add boundary condition to the correct surface(s)
        # self.boundary_condition(ids)
        return ids_map

    @abstractmethod
    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        raise NotImplementedError

    def boundary_condition(self, cad_surface_ids):
        if self.boundary_type == 'transmission':
            return []
        exec_cubit(f'group \"boundary:{self.boundary_type}\" add surface {cad_surface_ids[2:]}')
        # TODO add dict of non transmission surfaces and propagate them

    @classmethod
    def from_openmc_surface(cls, surface):
        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            return cls.from_openmc_surface_inner(surface)

    @classmethod
    @abstractmethod
    def from_openmc_surface_inner(cls, surface):
        raise NotImplementedError


class CADPlane(CADSurface, openmc.Plane):

    @staticmethod
    def lreverse(node):
        return "" if node.side == '-' else "reverse"

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            n = np.array([self.coefficients[k] for k in ('a', 'b', 'c')])
            distance = self.coefficients['d'] / np.linalg.norm(n)

            # Create cutter block larger than the world and rotate/translate it so
            # the +z plane of the block is coincident with this general plane
            max_extent = np.max(extents)
            exec_cubit(f"brick x {2*max_extent} y {2*max_extent} z {2*max_extent}" )
            ids_map = volume_id()
            exec_cubit(f"volume {{ { ids_map } }} move 0.0 0.0 {-max_extent}")

            nhat = n / np.linalg.norm(n)
            rhat = np.array([0.0, 0.0, 1.0])
            angle = math.degrees(math.acos(np.dot(nhat, rhat)))

            if not math.isclose(angle, 0.0, abs_tol=1e-6):
                rot_axis = np.cross(rhat, nhat)
                rot_axis /= np.linalg.norm(rot_axis)
                axis = f"{rot_axis[0]} {rot_axis[1]} {rot_axis[2]}"
                exec_cubit(f"Rotate volume {{ {ids_map} }} about 0 0 0 direction {axis} Angle {angle}")

            tvec = distance*nhat
            exec_cubit(f"volume {{ { ids_map } }} move {tvec[0]} {tvec[1]} {tvec[2]}")
            wid_map = volume_id()
            exec_cubit(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
            w_id = volume_id()
            # if positive half space we subtract the cutter block from the world
            if node.side != '-':
                exec_cubit(f"subtract volume {{ { ids_map } }} from volume {w_id}")
            # if negative half space we intersect the cutter block with the world
            else:
                exec_cubit(f"intersect volume {{ { ids_map } }} {w_id}")
        #TODO check what if there were multiple interactions, add strt, stp
        return np.array(volume_id())

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(a=plane.a, b=plane.b, c=plane.c, d=plane.d, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADXPlane(CADSurface, openmc.XPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            exec_cubit(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
            ids_map = volume_id()
            exec_cubit(f"section volume {{ {ids_map} }} with xplane offset {self.coefficients['x0']-off_center[0]} {self.reverse(node)}")
            surf_map[surf_id(node)] = ids_map
        return surf_map[surf_id(node)]

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(x0=plane.x0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADYPlane(CADSurface, openmc.YPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            exec_cubit(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
            id_map = volume_id()
            exec_cubit(f"section volume {{ {id_map} }} with yplane offset {self.coefficients['y0']-off_center[1]} {self.reverse(node)}")
            surf_map[surf_id(node)] = id_map
        return surf_map[surf_id(node)]

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(y0=plane.y0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADZPlane(CADSurface, openmc.ZPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            exec_cubit(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
            ids_map = volume_id()
            exec_cubit(f"section volume {{ {ids_map} }} with zplane offset {self.coefficients['z0']-off_center[2]} {self.reverse(node)}")
            surf_map[surf_id(node)] = ids_map
        return surf_map[surf_id(node)]

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(z0=plane.z0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)

class CADCylinder(CADSurface, openmc.Cylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        print('XCADCylinder to cubit surface')
        h = inner_world[2] if inner_world else extents[2]
        exec_cubit(f"cylinder height {h} radius {self.r}")
        ids_map = volume_id()
        if node.side != '-':
            wid_map = 0
            if inner_world:
                if hex:
                    exec_cubit(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                    wid_map = volume_id()
                    exec_cubit(f"rotate volume {{ {wid_map} }} about z angle 30")
                else:
                    exec_cubit(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                    wid_map = volume_id()
            else:
                exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                wid_map = volume_id()
            exec_cubit( f"subtract volume {{ { ids_map } }} from volume {{ { wid_map } }}" )
            rotate( wid_map, self.dx, self.dy, self.dz )
            move( wid_map, self.x0, self.y0, self.z0 )
            return wid_map
        rotate( ids_map,self.dx, self.dy, self.dz )
        move( ids_map,self.x0, self.y0, self.z0 )
        return ids_map

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, x0=cyl.x0, y0=cyl.y0, z0=cyl.z0, dx=cyl.dx, dy=cyl.dy, dz=cyl.dz,
                   boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)

class CADXCylinder(CADSurface, openmc.XCylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            h = inner_world[0] if inner_world else extents[0]
            exec_cubit( f"cylinder height {h} radius {self.r}")
            ids_map = volume_id()
            exec_cubit(f"rotate volume {{ {ids_map} }} about y angle 90")
            if node.side != '-':
                wid_map = 0
                if inner_world:
                    if hex:
                        exec_cubit(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                        wid_map = volume_id()
                        exec_cubit(f"rotate volume {{ {wid_map} }} about z angle 30")
                        exec_cubit(f"rotate volume {{ {wid_map} }} about y angle 90")
                    else:
                        exec_cubit(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                        wid_map = volume_id()
                else:
                    exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                    wid_map = volume_id()
                exec_cubit(f"subtract volume {{ { ids_map } }} from volume {{ { wid_map } }}")
            move(wid_map, 0, self.y0, self.z0)
            surf_map[surf_id(node)] = ids_map
        return surf_map[surf_id(node)]

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, y0=cyl.y0, z0=cyl.z0, boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)


class CADYCylinder(CADSurface, openmc.YCylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            h = inner_world[1] if inner_world else extents[1]
            exec_cubit( f"cylinder height {h} radius {self.r}")
            ids_map = volume_id()
            exec_cubit(f"rotate volume {{ {ids_map} }} about x angle 90")
            if node.side != '-':
                wid_map = 0
                if inner_world:
                    if hex:
                        exec_cubit(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                        wid_map = volume_id()
                        exec_cubit(f"rotate volume {{ {wid_map} }} about z angle 30")
                        exec_cubit(f"rotate volume {{ {wid_map} }} about x angle 90")
                    else:
                        exec_cubit(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                        wid_map = volume_id()
                else:
                    exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                    wid_map = volume_id()
                exec_cubit(f"subtract volume {{ { ids_map } }} from volume {{ { wid_map } }}")
            move(wid_map, self.x0, 0, self.z0)
            surf_map[surf_id(node)] = ids_map
        return surf_map[surf_id(node)]

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, x0=cyl.x0, z0=cyl.z0, boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)


class CADZCylinder(CADSurface, openmc.ZCylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        global surf_map
        if surf_id(node) not in surf_map:
            h = inner_world[2] if inner_world else extents[2]
            exec_cubit(f"cylinder height {h} radius {self.r}")
            ids_map = volume_id()
            if node.side != '-':
                if inner_world:
                    if hex:
                        exec_cubit(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                        wid_map = volume_id()
                        exec_cubit(f"rotate volume {{ {wid_map} }} about z angle 30")
                    else:
                        exec_cubit(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                        wid_map = volume_id()
                else:
                    exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                    wid_map = volume_id()
                exec_cubit(f"subtract volume {{ { ids_map } }} from volume {{ { wid_map } }} keep_tool")
                ids_map = wid_map
            move(ids_map, self.x0, self.y0, 0)
            surf_map[surf_id(node)] = ids_map
        return surf_map[surf_id(node)]

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, x0=cyl.x0, y0=cyl.y0, boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)


class CADSphere(CADSurface, openmc.Sphere):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        exec_cubit( f"sphere radius {self.r}")
        ids_map = volume_id()
        move(ids_map, self.x0, self.y0, self.z0)
        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            ids_map = volume_id()
        return ids_map

    @classmethod
    def from_openmc_surface_inner(cls, sphere):
        return cls(r=sphere.r, x0=sphere.x0, y0=sphere.y0, z0=sphere.z0, boundary_type=sphere.boundary_type, albedo=sphere.albedo, name=sphere.name, surface_id=sphere.id)

class CADCone(CADSurface):

    def to_cubit_surface(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        raise NotImplementedError('General Cones are not yet supported')

    @classmethod
    def from_openmc_surface(cls, surface):
        raise NotImplementedError('General Cones are not yet supported')

class CADXCone(CADSurface, openmc.XCone):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        exec_cubit( f"create frustum height {extents[0]} radius {math.sqrt(self.coefficients['r2'])*extents[0]} top 0")
        ids_map = volume_id()
        exec_cubit(f"body {{ {ids_map} }} move 0 0 -{extents[0]/2.0}")
        exec_cubit(f"body {{ {ids_map} }} copy reflect z")
        ids2 = volume_id()
        exec_cubit(f"unite volume {{ {ids_map} }}  {{ {ids2} }}")
        exec_cubit( f"rotate volume {{ {ids_map} }} about y angle 90")
        x0, y0, z0 = self.coefficients['x0'], self.coefficients['y0'], self.coefficients['z0']
        exec_cubit(f"body {{ {ids_map} }} move {x0} {y0} {z0}")

        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            ids_map = volume_id()
        return ids_map

    @classmethod
    def from_openmc_surface_inner(cls, surface):
        return cls(x0=surface.x0, y0=surface.y0, z0=surface.z0, r2=surface.r2, boundary_type=surface.boundary_type, albedo=surface.albedo, name=surface.name, surface_id=surface.id)


class CADYCone(CADSurface, openmc.YCone):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        exec_cubit( f"create frustum height {extents[1]} radius {math.sqrt(self.coefficients['r2'])*extents[1]} top 0")
        ids_map = volume_id()
        exec_cubit(f"body {{ {ids_map} }} move 0 0 -{extents[1]/2.0}")
        exec_cubit(f"body {{ {ids_map} }} copy reflect z")
        ids2 = volume_id()
        exec_cubit(f"unite volume {{ {ids_map} }}  {{ {ids2} }}")
        exec_cubit( f"rotate volume {{ {ids_map} }} about x angle 90")
        x0, y0, z0 = self.coefficients['x0'], self.coefficients['y0'], self.coefficients['z0']
        exec_cubit(f"body {{ {ids_map} }} move {x0} {y0} {z0}")

        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            ids_map = volume_id()
        return ids_map

    @classmethod
    def from_openmc_surface_inner(cls, surface):
        return cls(x0=surface.x0, y0=surface.y0, z0=surface.z0, r2=surface.r2, boundary_type=surface.boundary_type, albedo=surface.albedo, name=surface.name, surface_id=surface.id)


class CADZCone(CADSurface, openmc.ZCone):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        exec_cubit( f"create frustum height {extents[2]} radius {math.sqrt(self.coefficients['r2'])*extents[2]} top 0")
        ids_map = volume_id()
        exec_cubit(f"body {{ {ids_map} }} move 0 0 -{extents[2]/2.0}")
        exec_cubit(f"body {{ {ids_map} }} copy reflect z")
        ids2 = volume_id()
        exec_cubit(f"unite volume {{ {ids_map} }}  {{ {ids2} }}")
        x0, y0, z0 = self.coefficients['x0'], self.coefficients['y0'], self.coefficients['z0']
        exec_cubit(f"body {{ {ids_map} }} move {x0} {y0} {z0}")

        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            ids_map = volume_id()
        return ids_map

    @classmethod
    def from_openmc_surface_inner(cls, surface):
        return cls(x0=surface.x0, y0=surface.y0, z0=surface.z0, r2=surface.r2, boundary_type=surface.boundary_type, albedo=surface.albedo, name=surface.name, surface_id=surface.id)


class CADTorus(CADSurface):

    def check_coeffs(self):
        if self.b != self.c:
            raise ValueError("Only torri with constant minor radii are supported")

    @classmethod
    def from_openmc_surface_inner(cls, surface):
        return cls(x0=surface.x0, y0=surface.y0, z0=surface.z0, a=surface.a, b=surface.b, c=surface.c, boundary_type=surface.boundary_type, albedo=surface.albedo, name=surface.name, surface_id=surface.id)

class CADXTorus(CADTorus, openmc.XTorus):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        self.check_coeffs()
        
        exec_cubit( f"torus major radius {self.a} minor radius {self.b}" )
        ids_map = volume_id()
        exec_cubit( f"rotate volume {{ {ids_map} }} about y angle 90")
        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            move(wid_map, self.x0, self.y0, self.z0)
            ids_map = volume_id()
        else:
            move(ids_map, self.x0, self.y0, self.z0)
        return ids_map


class CADYTorus(CADTorus, openmc.YTorus):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        self.check_coeffs()
        
        exec_cubit( f"torus major radius {self.a} minor radius {self.b}" )
        ids_map = volume_id()
        exec_cubit( f"rotate volume {{ {ids_map} }} about x angle 90")
        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            move(wid_map, self.x0, self.y0, self.z0)
            ids_map = volume_id()
        else:
            move(ids_map, self.x0, self.y0, self.z0)
        return ids_map


class CADZTorus(CADTorus, openmc.ZTorus):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False, off_center=[0,0,0]):
        self.check_coeffs()
        
        exec_cubit( f"torus major radius {self.a} minor radius {self.b}" )
        ids_map = volume_id()
        if node.side != '-':
            exec_cubit( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
            wid_map = volume_id()
            exec_cubit(f"subtract volume {{ {ids_map} }} from volume {{ {wid_map} }}")
            move(wid_map, self.x0, self.y0, self.z0)
            ids_map = wid_map
        else:
            move(ids_map, self.x0, self.y0, self.z0)
        return ids_map


_CAD_SURFACES = [CADPlane, CADXPlane, CADYPlane, CADZPlane, CADCylinder, CADXCylinder, CADYCylinder, CADZCylinder, CADSphere, CADXCone, CADYCone, CADZCone, CADXTorus, CADYTorus, CADZTorus]

_CAD_SURFACE_DICTIONARY = {s._type: s for s in _CAD_SURFACES}
