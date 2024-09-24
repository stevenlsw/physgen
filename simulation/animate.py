import os
from typing import Optional
import argparse
from omegaconf import OmegaConf
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from imantics import Mask

from animate_utils import AnimateObj, AnimateImgSeg
from sim_utils import fit_circle_from_mask, list_to_numpy



class AnimateUni(AnimateObj):

    def __init__(self,
                 save_dir, 
                 mask_path,
                 obj_info: dict,
                 edge_list: Optional[np.array]=None,
                 init_velocity: Optional[dict]=None,
                 init_acc: Optional[dict]=None, 
                 gravity=980, 
                 ground_elasticity=0,
                 ground_friction=1,
                 num_steps=200, 
                 size=(512, 512), 
                 display=True, 
                 save_snapshot: Optional[bool]=False,
                 snapshot_frames: Optional[int]=16,
                 colormap: Optional[str]="tab20") -> None:
        
        
        super(AnimateUni, self).__init__(save_dir, size, gravity, display)
        self.num_steps = num_steps
        self.objs = {} # key is seg_id, value is pymunk shape
        mask_img = cv2.imread(mask_path, 0)
        self.mask_img = mask_img
        
        self.save_snapshot = save_snapshot
        if self.save_snapshot:
            self.snapshot_dir = os.path.join(save_dir, "snapshot")
            os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_frames = snapshot_frames
        self.colormap = colormap
        
        self.obj_info = obj_info
        
        for seg_id in self.obj_info:
            class_name =  self.obj_info[seg_id]["primitive"]
            density =  self.obj_info[seg_id]["density"]
            mass =  self.obj_info[seg_id]["mass"]
            elasticity =  self.obj_info[seg_id]["elasticity"]
            friction =  self.obj_info[seg_id]["friction"]
            if class_name == "polygon":
                mask = self.mask_img == seg_id
                polygons = Mask(mask).polygons()
                if len(polygons.points) > 1: # find the largest polygon
                    areas = []
                    for points in polygons.points:
                        points = points.reshape((-1, 1, 2)).astype(np.int32)
                        img = np.zeros((self.mask_img.shape[0], self.mask_img.shape[1], 3), dtype=np.uint8)
                        img = cv2.fillPoly(img, [points], color=[0, 255, 0])
                        mask = img[:, :, 1] > 0
                        area = np.count_nonzero(mask)
                        areas.append(area)
                    areas = np.array(areas)
                    largest_idx = np.argmax(areas)
                    points = polygons.points[largest_idx]
                else:
                    points = polygons.points[0]
                
                points = tuple(map(tuple, points))
                poly = self._create_poly(density, points, elasticity, friction)
                self.objs[seg_id] = poly
            
            elif class_name == "circle":
                mask = self.mask_img == int(seg_id)
                mask = (mask * 255).astype(np.uint8)
                center, radius = fit_circle_from_mask(mask)
                ball = self._create_ball(mass, radius, center, elasticity, friction)
                self.objs[seg_id] = ball
       
        if edge_list is not None:
            self._create_wall_segments(edge_list, ground_elasticity, ground_friction)
       
        self.init_velocity = init_velocity
        self.init_acc = init_acc
       
    def _create_wall_segments(self, edge_list, elasticity=0.05, friction=0.9):
        """Create a number of wall segments connecting the points"""
        for edge in edge_list:
            point_1, point_2 = edge
            v1 = pymunk.Vec2d(point_1[0], point_1[1])
            v2 = pymunk.Vec2d(point_2[0], point_2[1])
            wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            wall_shape = pymunk.Segment(wall_body, v1, v2, 0.0)
            wall_shape.collision_type = 0
            wall_shape.elasticity = elasticity
            wall_shape.friction = friction
            self._space.add(wall_body, wall_shape)
            
    def _create_ball(self, mass, radius, position, elasticity=0.5, friction=0.4):
        """
        Create a ball defined by mass, radius and position
        """
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = elasticity
        shape.friction = friction
        self._space.add(body, shape)
        return shape 
        
    def _create_poly(self, density, points, elasticity=0.5, friction=0.4, collision_type=0):
        """
        Create a poly defined by density, points
        """
        body = pymunk.Body()
        shape = pymunk.Poly(body, points)
        shape.density = density
        shape.elasticity = elasticity
        shape.friction = friction
        shape.collision_type = collision_type
        self._space.add(body, shape)
        return shape 
      
   
    def _draw_objects(self) -> None:
        jet = plt.get_cmap(self.colormap) 
        cNorm  = colors.Normalize(vmin=0, vmax=len(self.objs))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
        for seg_id in self.objs:
            shape = self.objs[seg_id]
            color = scalarMap.to_rgba(int(seg_id))
            color = (color[0]*255, color[1]*255, color[2]*255, 255)
            if isinstance(shape, pymunk.Circle):
                self.draw_ball(shape, color)
            elif isinstance(shape, pymunk.Poly):
                self.draw_poly(shape, color)
        for shape in self._space.shapes:
            if isinstance(shape, pymunk.Segment):
                self.draw_wall(shape)
            
    def draw_ball(self, ball, color=(0, 0, 255, 255)):
        body = ball.body
        v = body.position + ball.offset.cpvrotate(body.rotation_vector)
        r = ball.radius
        pygame.draw.circle(self._screen, pygame.Color(color), v, int(r), 0)

    def draw_wall(self, wall, color=(252, 3, 169, 255), width=3):
        body = wall.body
        pv1 = body.position + wall.a.cpvrotate(body.rotation_vector)
        pv2 = body.position + wall.b.cpvrotate(body.rotation_vector)
        pygame.draw.lines(self._screen, pygame.Color(color), False, [pv1, pv2], width=width)

    def draw_poly(self, poly, color=(0, 255, 0, 255)):
        body = poly.body
        ps = [p.rotated(body.angle) + body.position for p in poly.get_vertices()]
        ps.append(ps[0])
        pygame.draw.polygon(self._screen, pygame.Color(color), ps)
            
    def get_transform(self):
        # get current timestep objs state transformation
        """
        rotation_matrix: np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        in Kornia, the rotation matrix created by K.geometry.transform.get_rotation_matrix2d should be transpose,
        equivalent to use -angle in record
        """
        state = {}
        keys = list(self.objs.keys())
        keys = sorted(keys, key=lambda x: int(x))
        for seg_id in keys:
            shape = self.objs[seg_id]
            if isinstance(shape, pymunk.Poly):
                ps = [p.rotated(shape.body.angle) + shape.body.position for p in shape.get_vertices()]
                ps = np.array(ps)
                center = np.mean(ps, axis=0)
                angle = shape.body.angle

            elif isinstance(shape, pymunk.Circle):
                center = shape.body.position
                angle = shape.body.angle     
                 
            state[seg_id] = (center, angle)
        return state
       
    def init_condition(self, init_velocity, init_acc):
        for seg_id in self.objs:
            shape = self.objs[seg_id]
            query_seg_id = seg_id
            if init_velocity is not None and query_seg_id in init_velocity:
                ins_init_vel = list(init_velocity[query_seg_id]) if not isinstance(init_velocity[query_seg_id], list) else init_velocity[query_seg_id]
                shape.body.velocity = ins_init_vel
            if init_acc is not None and query_seg_id in init_acc:
                ins_init_acc = init_acc[query_seg_id]
                shape.body.apply_impulse_at_local_point((ins_init_acc[0] * shape.body.mass, ins_init_acc[1] * shape.body.mass), (0, 0))
               
    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        num_steps = self.num_steps // self._physics_steps_per_frame
        count = 0
        self.init_condition(self.init_velocity, self.init_acc)
        if self.save_snapshot:
            # save snapshot of the simulation of #animate frames
            snapshot_indices = np.arange(num_steps)[::num_steps//(self.snapshot_frames)][:self.snapshot_frames]
            snapshot_indices[0] = 0
            save_path = os.path.join(self.snapshot_dir, "snapshot_{:03d}.png")
            self._process_events()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            pygame.image.save(self._screen, save_path.format(0))
            
        while self._running and count < num_steps:
            # Progress time forward
            count += 1
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
            state = self.get_transform()
            self.history.append(state)
            if self.display:
                self._process_events()
                self._clear_screen()
                self._draw_objects()
                pygame.display.flip()
                if self.save_snapshot and count in snapshot_indices:
                    index = np.where(snapshot_indices==count)[0].item()
                    pygame.image.save(self._screen, save_path.format(index))
                
                self._clock.tick(50)
                pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
        self.save_state()
    

def main(args, data_root, save_root):
    save_dir =  os.path.join(save_root, args.cat.lower())
    os.makedirs(save_dir, exist_ok=True)
    
    data_dir = os.path.join(data_root, args.cat)
    mask_path=os.path.join(data_dir, "mask.png")
            
    anim = AnimateUni(
        mask_path=mask_path, save_dir=save_dir,
        obj_info=args.obj_info, edge_list=args.edge_list,
        init_velocity=args.init_velocity, init_acc=args.init_acc,
        num_steps=args.num_steps,
        gravity=args.gravity, 
        ground_elasticity=getattr(args, "ground_elasticity", 0),
        ground_friction=getattr(args, "ground_friction", 1),
        size=args.size, display=args.display, save_snapshot=args.save_snapshot, snapshot_frames=args.animation_frames)
    anim.run()
    
    history_path = os.path.join(save_dir, "history.pkl")
    animation_frames = getattr(args, "animation_frames", 16)
    replay = AnimateImgSeg(data_dir=data_dir, save_dir=save_dir, history_path=history_path, animate_frames=animation_frames)
    replay.record()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--save_root", type=str, default="../outputs")
    parser.add_argument("--config", type=str, default="../data/pool/sim.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config = list_to_numpy(config)
    main(config, data_root=args.data_root, save_root=args.save_root)