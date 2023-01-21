import json
from json.encoder import INFINITY

import numpy as np
import cv2
import collections

def vector_to_xyz(vec):
    return {"x": vec.x, "y": vec.y, "z": vec.z}

def green(list):
    return list[1]
def height(list):
    return list[4]
def width(list):
    return list[3]

def write_json_bb_opt(filename, i_str, image = False):
    img = cv2.imread(f"{filename}.png")
    img_depth = cv2.imread(f'{i_str}_depth.png')
    B_dep, G_dep, R_dep = cv2.split(img_depth)
    Depth = 1000*(R_dep+G_dep*256+B_dep*256**2)/(256**3-1)
    B, G, R = cv2.split(img)
    Gr = set(G.flatten())
    Bl = set(B.flatten())
    list_of_retangles=[]
    color_veh = (0,165,255)
    color_ped = (255,165,0)
    thickness = 2
    num_veh = 0
    for i in Gr:
        for k in Bl:
            pixel = np.where((G == i) &(B == k))
            if (len(pixel[0])== 0) and (len(pixel[1]) == 0):
                continue
            j = R[pixel[0][0], pixel[1][0]]
            if j == 4 or j == 10:
                veh_ped_dict = {}
                mask = (R  == j) & (G == i) & (B == k)
                sum_column = mask.sum(axis = 0)
                sum_row = mask.sum(axis = 1)
                column = np.where(sum_column > 0)
                row = np.where(sum_row > 0)
                min_x = int(np.min(column[0]))
                max_x = int(np.max(column[0]))
                min_y = int(np.min(row[0]))
                max_y = int(np.max(row[0]))
                if R[pixel[0][0], pixel[1][0]] == 4:
                    veh_ped_dict['pedestian'] = int(i)
                elif R[pixel[0][0],  pixel[1][0]] == 10:
                    veh_ped_dict['vehicle'] = int(i)
                    num_veh += 1
                veh_ped_dict['min_point'] = (min_x, min_y)
                veh_ped_dict['max_point'] = (max_x, max_y)
                if min_x == max_x or min_y == max_y:
                    pre_distance_veh = 100
                else:
                    pre_distance_veh = np.min(Depth[mask])
                veh_ped_dict['distance'] = pre_distance_veh
                list_of_retangles.append(veh_ped_dict)
    file_bb = open(f'{i_str}_boundingbox.json', 'w')
    if list_of_retangles == []:
        print('There is no object near the ego vehicle. The bounding box image is the same like the rgb image.')
    else:
        json.dump(list_of_retangles, file_bb)
        file_bb.close()
        json_file = json.load(open(f'{i_str}_boundingbox.json'))
        if image == True:
            drawboundingboxes(json_file, i_str, color_veh, color_ped, thickness, num_veh)
      

def write_json_bb(filename, measurement_heigth, measurement_width , i_str, image = False):
    c_vehicles = collections.Counter()
    c_pedestians = collections.Counter()
    img = cv2.imread(f"{filename}.png")
    img_depth = cv2.imread(f"{i_str}_depth.png")
    id_vehicle = []
    id_pedestians = []
    z = 0
    v = 0
    list_of_retangles = []
    color_veh = (0,165,255) 
    color_ped = (255,165,0)
    thickness = 2
    for y in range(measurement_heigth):
        for x in range(measurement_width):
            r = img.item(y,x,2)
            
            if r == 4:
                (b,g,r) = img[y,x]
                id_pedestians.append((r,g,b,x,y))
            elif r == 10:
                (b,g,r) = img[y,x]
                id_vehicle.append((r,g,b,x,y))
                
    id_vehicle.sort(key = green)
    id_pedestians.sort(key = green)
    for i in range(len(id_vehicle)):
        k = id_vehicle[i][1]
        c_vehicles[k]+=1
    for j in range(len(id_pedestians)):
        n = id_pedestians[j][1]
        c_pedestians[n]+=1
    
   
    for veh in c_vehicles:
        pre_distance_veh = np.Infinity
        veh_ped_dict = {}
        id_vehicle_single = id_vehicle[z:z+c_vehicles[veh]]
        z += c_vehicles[veh]
        for l in range(len(id_vehicle_single)):
            (bl, gr, re) = img_depth[id_vehicle_single[l][4],id_vehicle_single[l][3]]
            distance = 1000*(re + gr * 256 +bl * 256**2)/(256**3-1)
            if distance < pre_distance_veh:
                pre_distance_veh = distance
        veh_ped_dict['vehicle'] = int(veh)
        id_vehicle_single.sort(key = width)
        min_x = id_vehicle_single[0][3]
        max_x = id_vehicle_single[c_vehicles[veh]-1][3]
        id_vehicle_single.sort(key = height)
        min_y = id_vehicle_single[0][4]
        max_y = id_vehicle_single[c_vehicles[veh]-1][4]
        veh_ped_dict['min_point'] = (min_x, min_y)
        veh_ped_dict['max_point'] = (max_x, max_y)
        veh_ped_dict['distance'] = pre_distance_veh
        list_of_retangles.append(veh_ped_dict)
    num_veh = len(list_of_retangles)
    for ped in c_pedestians:
        pre_distance_ped = np.Infinity
        veh_ped_dict = {}
        id_pedestian_single = id_pedestians[v:v+c_pedestians[ped]]
        v += c_pedestians[ped]
        for m in range(len(id_pedestian_single)):
            (bl, gr, re) = img_depth[id_pedestian_single[m][4],id_pedestian_single[m][3]]
            distance = 1000*(re + gr * 256 +bl * 256**2)/(256**3-1)
            if distance < pre_distance_ped:
                pre_distance_ped = distance
       

        veh_ped_dict['pedestian'] = int(ped)
        id_pedestian_single.sort(key = width)
        min_x = id_pedestian_single[0][3]
        max_x = id_pedestian_single[c_pedestians[ped]-1][3]
        id_pedestian_single.sort(key = height)
        min_y = id_pedestian_single[0][4]
        max_y = id_pedestian_single[c_pedestians[ped]-1][4]
        veh_ped_dict['min_point'] = (min_x, min_y)
        veh_ped_dict['max_point'] = (max_x, max_y)    
        veh_ped_dict['distance'] = pre_distance_ped     
        list_of_retangles.append(veh_ped_dict)
    file_bb = open(f'{i_str}_boundingbox.json', 'w')
    if list_of_retangles == []:
        print('There is no object near the ego vehicle. The bounding box image is the same like the rgb image.')
    else:
        json.dump(list_of_retangles, file_bb)
        file_bb.close()
        json_file = json.load(open(f'{i_str}_boundingbox.json'))
        if image == True:
            drawboundingboxes(json_file, i_str, color_veh, color_ped, thickness, num_veh)
def drawboundingboxes(jsonfile, img_num, color_veh, color_ped, thickness, num_veh):
    img = cv2.imread(f'{img_num}_rgb.png')
    for i in range(len(jsonfile)):
        min_point = jsonfile[i]['min_point']
        max_point = jsonfile[i]['max_point']
        if jsonfile[i]['distance'] < 85.0:
            if i < num_veh:
                img = cv2.rectangle(img,min_point, max_point,  color_veh, thickness)
            else: 
                img = cv2.rectangle(img,min_point, max_point, color_ped, thickness) 
    cv2.imwrite( f'{img_num}_rgb_bb.png',img)

def panoptic_segmentation(i_str):
    img = cv2.imread(f'{i_str}_instance.png')
    instance_seg = np.zeros_like(img)

    panoptic_cs = cv2.imread(f'{i_str}_semantic_cs.png')
    

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            r = img.item(y, x, 2)
            if (r == 4 or r == 10):
                instance_seg[y, x, :] = img[y, x, :]
                panoptic_cs[y, x, :] = img[y, x, :]
                                
    cv2.imwrite(f"{i_str}_instance_proper.png", instance_seg)
    cv2.imwrite(f'{i_str}_panoptic_cs.png', panoptic_cs)

    return