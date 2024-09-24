import os
import torch
from typing import List
import numpy as np
import pickle
import pygame
import pymunk
import pymunk.pygame_util

import kornia as K
from sim_utils import writing_video, prep_data, composite_trans


class AnimateObj(object):

    def __init__(self, save_dir, size=(512,512), gravity=980, display=True) -> None:
        self.size = size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.history = [] # record state
        
        # Space
        self._space = pymunk.Space()        
        # Physics
        # Time step
        self._space.gravity = (0.0, gravity) # positive_y_is_down
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1 # larger, faster
        self.history = []
        # pygame
        self.display = display
        if self.display:
            pygame.init()
            self._screen = pygame.display.set_mode(size)
            self._clock = pygame.time.Clock()
            self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Execution control and time until the next ball spawns
        self._running = True

    def run(self) -> None:
        """Custom implement func
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
                
            state = self.get_transform()
            self.history.append(state)
            if self.display:
                self._process_events()
                self.draw()
                # self._clear_screen()
                # self._draw_objects()
                pygame.display.flip()
                # Delay fixed time between frames
                self._clock.tick(50)
                pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
        self.save_state()
        

    def get_transform(self, verbose=False) -> List[int]:
        # custom func
        center = self.ball.body.position
        angle = self.poly.body.angle
        if verbose:
            print(center, angle)
        return center, angle
        
    def save_state(self):
         # save self.state
        with open(os.path.join(self.save_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(self.history, f)
        
    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            
    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)


class AnimateImgSeg(object):
    # Animate Img from Segmentation
    def __init__(self, data_dir, save_dir, history_path, animate_frames=16):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.img, self.mask_img, inpaint_img = prep_data(data_dir)
    
        with open(history_path, 'rb') as f:
            history = pickle.load(f)     
        self.animate_frames = animate_frames
        self.history = history[::len(history)//(self.animate_frames)][:animate_frames]
        assert len(self.history) == self.animate_frames
        
        active_keys = list(self.history[0].keys())
        self.active_keys = sorted(active_keys, key=lambda x: int(x))
        for seg_id in np.unique(self.mask_img):
            if seg_id == 0: # background
                continue
            if seg_id not in active_keys:
                mask = self.mask_img == int(seg_id)
                inpaint_img[mask] = self.img[mask]
                self.mask_img[mask] = 0 # set as background
        self.inpaint_img = inpaint_img # inpaint_img is the background
        
        init_centers = []
        for seg_id in active_keys:
            if seg_id == "0":
                raise ValueError("seg_id should not be 0")
            mask = self.mask_img == int(seg_id)
            init_center = np.mean(np.argwhere(mask), axis=0)
            init_center = np.array([init_center[1], init_center[0]]) # center (y,x) to (x,y)
            init_centers.append(init_center)
        init_centers = np.stack(init_centers, axis=0) # (num_objs, 2)
        self.init_centers = init_centers.astype(np.float32)
                
    def record(self):
        H, W = self.img.shape[:2]
        masked_src_imgs = []
        for seg_id in self.active_keys:
            mask = self.mask_img == int(seg_id)
            src_img = torch.from_numpy(self.img * mask[:, :, None]).permute(2, 0, 1).float() # (3, H, W)
            masked_src_imgs.append(src_img)
        masked_src_imgs = torch.stack(masked_src_imgs, dim=0) # tensor (num_objs, 3, H, W)
        init_centers = torch.from_numpy(self.init_centers).float()# tensor (num_objs, 2)
        
        imgs = []
        msk_list = []
        trans_list = [] # (animate_frames [dict{seg_id: Trans(2, 3)}]
        for i in range(len(self.history)):
            if i == 0: # use original image
                trans = []
                imgs.append((self.img*255).astype(np.uint8))
                # active segmentation mask
                seg_mask = np.zeros_like(self.mask_img)
                for seg_id in self.active_keys:
                    seg_mask[self.mask_img == int(seg_id)] = seg_id
                    trans.append(torch.eye(3)[:2, :]) # (2, 3)
                msk_list.append(seg_mask)
                trans = torch.stack(trans, dim=0) # (num_objs, 2, 3)
                trans_list.append(trans)
            else:
                history = self.history[i] # dict of seg_id: (center, angle)
                centers, scales, angles = [], [], []
                for seg_id in self.active_keys:
                    center, angle = history[seg_id]
                                        
                    # the rotation matrix used in K is pymunk rotation transpose, thus use -angle
                    angle = -angle
                    
                    center = torch.tensor([center]).float() # (1, 2)
                    angle = torch.tensor([angle/np.pi * 180]).float() # [1]
                    scale = torch.ones(1, 2).float()
                    centers.append(center)
                    scales.append(scale)
                    angles.append(angle)
                centers = torch.cat(centers, dim=0) # (num_objs, 2)
                scales = torch.cat(scales, dim=0) # (num_objs, 2)
                angles = torch.cat(angles, dim=0) # (num_objs)

                trans = K.geometry.transform.get_rotation_matrix2d(init_centers, angles, scales)
                trans[:, :, 2] += centers - init_centers # (num_objs, 2, 3)
                
                active_list = list(map(int, self.active_keys))
                final_frame, seg_mask = composite_trans(masked_src_imgs, trans, self.inpaint_img, active_list)
                
                imgs.append(final_frame)
                msk_list.append(seg_mask)
                trans_list.append(trans) # (num_objs, 2, 3)
    
        imgs = np.stack(imgs, axis=0) # (animate_frames, H, W, 3)
        writing_video(imgs[..., ::-1], os.path.join(self.save_dir, 'composite.mp4'), frame_rate=7)
        
        msk_list = np.stack(msk_list, axis=0) # (animate_frames, H, W)
        msk_list = torch.from_numpy(msk_list)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        trans_list = np.stack(trans_list, axis=0) # (animate_frames, num_objs, 2, 3)
        trans_list = torch.from_numpy(trans_list)
        assert len(imgs) == len(msk_list) == len(trans_list)

        torch.save(msk_list, os.path.join(self.save_dir,'mask_video.pt')) # foreground objs segmentation mask
        torch.save(imgs, os.path.join(self.save_dir,'composite.pt'))
        torch.save(trans_list, os.path.join(self.save_dir, "trans_list.pt"))