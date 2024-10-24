import argparse
import cv2
import numpy as np
import os
import json
import matplotlib
import matplotlib.pyplot as plt


def vis_seg(mask_img, text=False):
    norm = matplotlib.colors.Normalize(vmin=np.min(mask_img), vmax=np.max(mask_img))
    norm_segmentation_map = norm(mask_img)
    cmap = "tab20"
    colormap = plt.get_cmap(cmap)
    colored_segmentation_map = colormap(norm_segmentation_map)
    colored_segmentation_map = (colored_segmentation_map[:, :, :3] * 255).astype(np.uint8)
    colored_segmentation_map[mask_img == 0] = [255, 255, 255]
    if text:
        unique_seg_ids = np.unique(mask_img)
        unique_seg_ids = unique_seg_ids[unique_seg_ids != 0]
        for seg_id in unique_seg_ids:
            mask_indices = np.where(mask_img == seg_id)
            if len(mask_indices[0]) > 0:
                center_y = int(np.mean(mask_indices[0]))
                center_x = int(np.mean(mask_indices[1]))
                cv2.putText(
                    colored_segmentation_map, 
                    str(seg_id), 
                    (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 0),
                    2, 
                    cv2.LINE_AA
                )
    return colored_segmentation_map


def refine_segmentation_mask(mask, seg_mask, value, min_size):
    """
    Refine a segmentation mask by removing small connected components.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    # Filter out small connected components
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            seg_mask[labels == label] = value
            value = value + 1

    return seg_mask, value


def find_neighbor_segmentation_classes(segmentation_map, class_id):
    """
    Find the neighboring segmentation classes of a specified class in a segmentation map.
    """
    class_mask = np.uint8(segmentation_map == class_id)
    dilated_mask = cv2.dilate(class_mask, np.ones((3, 3), np.uint8), iterations=3)
    neighbor_class_ids = np.unique(segmentation_map[dilated_mask > 0])
    neighbor_class_ids = neighbor_class_ids[neighbor_class_ids != class_id]
    return list(neighbor_class_ids)



def compute_angle_to_direction(normal, direction):
    """
    Compute the angle (in degrees) between a surface normal and a direction vector
    """
    dot_product = np.dot(normal, direction)
    norm_normal = np.linalg.norm(normal, axis=-1)
    norm_direction = np.linalg.norm(direction)
    cos_theta = dot_product / (norm_normal * norm_direction)
    return np.abs(cos_theta)


def normal_planes(normals):
    '''
        normal coordinate:
        y
        |
        o -- x
        /
        z
    '''
    upward_direction = np.array([0, -1, 0])
    downward_direction = np.array([0, 1, 0])
    leftward_direction = np.array([-1, 0, 0])
    rightward_direction = np.array([1, 0, 0])
    forward_direction = np.array([0, 0, 1])
    inward_direction = np.array([0, 0, -1])

    # Compute the angles between each pixel's normal and the 6 directions
    angle_leftward = compute_angle_to_direction(normals, leftward_direction)
    angle_rightward = compute_angle_to_direction(normals, rightward_direction)
    angle_X = np.maximum(angle_leftward, angle_rightward).mean()
    
    angle_upward = compute_angle_to_direction(normals, upward_direction)
    angle_downward = compute_angle_to_direction(normals, downward_direction)
    angle_Y = np.maximum(angle_upward, angle_downward).mean()
    
    angle_forward = compute_angle_to_direction(normals, forward_direction)
    angle_inward = compute_angle_to_direction(normals, inward_direction)
    angle_Z = np.maximum(angle_forward, angle_inward).mean()
    
    angles = np.array([angle_X, angle_Y, angle_Z])
    index = np.argmax(angles)
    if index == 0:
        return "X"
    elif index == 1:
        return "Y"
    else:
        return "Z"
    

def find_upper_contour(binary_mask):
    binary_mask = binary_mask.astype(np.uint8) 
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    
    start_index = np.where((contour[:, 0] == leftmost).all(axis=1))[0][0]
    end_index = np.where((contour[:, 0] == rightmost).all(axis=1))[0][0]

    if start_index > end_index:
        start_index, end_index = end_index, start_index

    subarray1 = contour[start_index:end_index+1]
    subarray2 = np.concatenate((contour[end_index:], contour[:start_index+1]))
    if subarray1[:, 0, 1].mean() < subarray2[:, 0, 1].mean():
        upper_contour = subarray1
    else:
        upper_contour = subarray2
    
    start_x, end_x = upper_contour[:, 0, 0].min(), upper_contour[:, 0, 0].max()
    start_x, end_x = int(start_x), int(end_x)
    y = upper_contour[:, 0, 1].mean()
    y = int(y)
    edge = [(start_x, y), (end_x, y)]
    return edge
    

def is_mask_truncated(mask):
    if np.any(mask[0, :] == 1) or np.any(mask[-1, :] == 1):
        return True
    if np.any(mask[:, 0] == 1) or np.any(mask[:, -1] == 1):
        return True
    return False

def draw_edge_on_image(image, edges, color=(0, 255, 0), thickness=2):
    output_image = image.copy()
    for edge in edges:
        for i in range(1, len(edge)):
            start_point = (edge[i-1][0], edge[i-1][1])
            end_point = (edge[i][0], edge[i][1])
            cv2.line(output_image, start_point, end_point, color, thickness)

    return output_image



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="../data/pig_ball", help="input path")
    parser.add_argument("--output", type=str, default=None, help="output directory")
    parser.add_argument("--vis_edge", action="store_true", help="visualize edges")
    
    args = parser.parse_args()
    
    output = args.output
    if output is None:
        output = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    os.makedirs(output, exist_ok=True)
        
    seg_path = os.path.join(args.input, "intermediate", 'mask.png')
    seg_mask = cv2.imread(seg_path, 0)
        
    depth_path = os.path.join(args.input, 'depth.npy')
    depth = np.load(depth_path)
    
    normal_path = os.path.join(args.input, 'normal.npy')
    normal = np.load(normal_path)
    
    seg_info_path = os.path.join(args.input, "intermediate", 'mask.json')
    with open(seg_info_path, 'r') as f:
        seg_info = json.load(f)
     
    seg_ids = np.unique(seg_mask)
     
    obj_infos = seg_info
    for seg_id in seg_ids:
        
        mask = seg_mask == seg_id
        seg_depth = depth[mask]
            
        min_depth = np.percentile(seg_depth, 5) 
        max_depth = np.percentile(seg_depth, 95)
        
        obj_infos[seg_id]["depth"] =  [min_depth, max_depth]
            
    movable_seg_ids = [seg_id for seg_id in seg_ids if obj_infos[seg_id]['movable']] 
    fg_seg_mask = np.zeros_like(seg_mask) 
    fg_truncated_mask = np.zeros_like(seg_mask)
    edges = []
    for seg_id in movable_seg_ids:
        mask = seg_mask == seg_id
        if mask.sum() < 100: # ignore small objects
            seg_mask[mask] = 0
            continue
        if is_mask_truncated(mask):
            fg_truncated_mask[mask] = seg_id
        else:
            fg_seg_mask[mask] = seg_id
    
    seg_mask_path = os.path.join(output, 'mask.png') # save the foreground mask as final mask
    fg_mask = fg_seg_mask + fg_truncated_mask
    cv2.imwrite(seg_mask_path, fg_mask)
    
    vis_seg_mask = vis_seg(fg_seg_mask, text=True)
    vis_save_path = os.path.join(output, "intermediate", 'fg_mask_vis.png')
    cv2.imwrite(vis_save_path, vis_seg_mask) 

    for seg_id in np.unique(fg_truncated_mask):
        if seg_id == 0:
            continue
        mask = fg_truncated_mask == seg_id
        points = cv2.findNonZero((mask> 0).astype(np.uint8))

        x, y, w, h = cv2.boundingRect(points)
                         
        top_edge = [[x, y], [x + w, y]]
        left_edge = [[x, y], [x, y + h]]
        right_edge = [[x + w, y], [x + w, y + h]]
        edges.extend([right_edge, left_edge, top_edge])         
                             
    bg_seg_mask = np.zeros_like(seg_mask) 
    value = 1
    nonmovable_seg_ids = [seg_id for seg_id in seg_ids if not obj_infos[seg_id]['movable']] 
    for seg_id in nonmovable_seg_ids:
        seg_area_ratio = np.sum(seg_mask == seg_id) / seg_mask.size
        if seg_id == 0 or seg_info[seg_id]['label'] == "background" or seg_area_ratio > 0.5:
            depth_threshold = np.array([obj_infos[seg_id]["depth"][0] for seg_id in movable_seg_ids]).min()
            mask = np.logical_and(seg_mask == seg_id, depth <= depth_threshold)
            min_size = int(500 / (512 * 512) * mask.size)
            bg_seg_mask, value = refine_segmentation_mask(mask, bg_seg_mask, value, min_size=min_size)
        
        else:
            neighbor_seg_ids = find_neighbor_segmentation_classes(seg_mask, seg_id)
            neighbor_seg_ids = [seg_id for seg_id in neighbor_seg_ids if seg_id in movable_seg_ids]
            if len(neighbor_seg_ids) == 0:
                depth_array = np.array([obj_infos[seg_id]["depth"][1] for seg_id in movable_seg_ids])
            else:
                depth_array = np.array([obj_infos[seg_id]["depth"][1] for seg_id in neighbor_seg_ids])
            
            min_threshold = depth_array.min()
            max_threshold = depth_array.max()
         
            min_size = int(500 / (512 * 512) * mask.size)
            mask = np.logical_and(seg_mask == seg_id, depth <= min_threshold)
            old_value = value
            bg_seg_mask, value = refine_segmentation_mask(mask, bg_seg_mask, value, min_size=min_size)
            
            if value == old_value:
                mask = np.logical_and(seg_mask == seg_id, min_threshold < depth)
                mask = np.logical_and(mask, depth <= max_threshold)
                bg_seg_mask, value = refine_segmentation_mask(mask, bg_seg_mask, value, min_size=min_size)
            
    seg_save_path = os.path.join(output, 'intermediate', 'bg_mask.png')
    cv2.imwrite(seg_save_path, bg_seg_mask)
    
    vis_seg_mask = vis_seg(bg_seg_mask)
    vis_seg_path = os.path.join(output, 'intermediate', 'bg_mask_vis.png')
    cv2.imwrite(vis_seg_path, vis_seg_mask)
    
    seg_ids = np.unique(bg_seg_mask)
    edge_map = np.zeros_like(bg_seg_mask)
    for seg_id in seg_ids:
        if seg_id == 0:
            continue
        else:
            mask = (bg_seg_mask == seg_id)
            normals = normal[mask]
            axis = normal_planes(normals)
            if axis == "X" or axis == "Z":
                points = cv2.findNonZero((mask> 0).astype(np.uint8))

                if points is not None:
                    x, y, w, h = cv2.boundingRect(points)
                    
                    if axis == "X":
                        right_col = x + w - 1
                        edge = [[right_col, y], [right_col, y + h]]
                        edges.append(edge)
                    else:  # Z
                        top_edge = [[x, y], [x + w, y]]
                        left_edge = [[x, y], [x, y + h]]
                        right_edge = [[x + w, y], [x + w, y + h]]
                        edges.extend([right_edge, left_edge, top_edge])         
                   
            elif axis == "Y":
                edge= find_upper_contour(mask)
                edges.append(edge)
    if args.vis_edge:
        img_path = os.path.join(args.input, 'original.png')
        img = cv2.imread(img_path)
        vis_edge_mask = draw_edge_on_image(img, edges, color=(0, 0, 255))
        vis_edge_save_path = os.path.join(output, 'intermediate', 'edge_vis.png')
        cv2.imwrite(vis_edge_save_path, vis_edge_mask)
    
    with open(os.path.join(output, 'edges.json'), 'w') as f:
        json.dump(edges, f)
    print("Done!")
           
        
    
    
        