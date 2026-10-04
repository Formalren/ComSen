from .utils import EventObject
from .components.Action import BaseAction
import math
import time
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from abc import ABC, abstractmethod

import threading
class BaseAgent(ABC):

    def __init__(self, controller: Controller, scene="FloorPlan203", 
                 visibilityDistance=1.5, gridSize=0.1, fieldOfView=90,platform_type="GPU"):
        
        
        self.scene = scene
        self.visibilityDistance = visibilityDistance
        self.gridSize = gridSize
        self.fieldOfView = fieldOfView
        if platform_type=="GPU":
            controller.reset(
                # platform=CloudRendering,
                snapToGrid=False,
                quality='Medium',
                agentMode="default",
                massThreshold=None,
                scene=scene,
                visibilityDistance=visibilityDistance,
                # gridSize=gridSize,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                width=800,
                height=450,
                fieldOfView=fieldOfView,
            )
        else: 
            controller.reset(
                snapToGrid=False,
                quality='Medium',
                agentMode="default",
                massThreshold=None,
                scene=scene,
                visibilityDistance=visibilityDistance,
                # gridSize=gridSize,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                width=800,
                height=450,
                fieldOfView=fieldOfView,
            )   
        
        self.controller = controller
        self.eventobject = EventObject()
        self.step_count = 0
        self.last_action = "INIT"
        self.mermory = []
        self.action = BaseAction()
        self.legal_location = {}
        # self.arm_reset()
        self.update_event()

    @abstractmethod
    def predict_next_action(self):
        pass

    def log_step_time_action(self, msg):
        print(f"time: {round(time.time())} step count: {self.step_count} setp: {self.controller.last_event.metadata}, action: {msg}")

    def loop(self):
        pass

    def update_event(self):
        pass
        
    def update_legal_location(self):
        visible_objects = []
        item_names, items = self.eventobject.get_visible_objects(self.controller.last_event)
        for item_name, item in zip(item_names, items):
            volume = self.eventobject.get_item_volume(self.controller.last_event, item_name)
            if volume <= 0.1:
                if item["distance"] <= 1.5:
                    visible_objects.append(item["name"])
            elif volume <= 0.5:
                if item["distance"] <= 2.5:
                    visible_objects.append(item["name"])
            elif volume <= 1:
                if item["distance"] <= 5.0:
                    visible_objects.append(item["name"])
            else:
                if item["distance"] <= 10.0:
                    visible_objects.append(item["name"])
        for item_name in visible_objects:
            if item_name not in self.legal_location.keys():
                self.legal_location[item_name] = 1
            else:
                self.legal_location[item_name] += 1

    def get_agent_position(self):
        return self.controller.last_event.metadata['agent']['position']
    
    def get_agent_rotation(self):
        return self.controller.last_event.metadata['agent']['rotation']
    
    def get_agent_horizon(self):
        return self.controller.last_event.metadata['agent']['cameraHorizon']
    
    def get_camera_position(self):
        return self.controller.last_event.metadata['cameraPosition']

    def get_camera_rotation(self):
        return self.controller.last_event.pose_discrete[3]

    def save_frame(self, kargs={}, prefix_save_path="./data/item_image"):
        import os
        if prefix_save_path != "./data/item_image":
            path = prefix_save_path
        else:
            path = os.path.join(prefix_save_path, self.scene)
        if not os.path.exists(path):
            os.makedirs(path)
        
        image_name = ""
        for key in kargs.keys():
            if key != "third_party_camera_frames" and key != "no_agent_view":
                image_name += f"_{kargs[key]}"
                
        if "third_party_camera_frames" in kargs.keys():
            image = Image.fromarray(self.controller.last_event.third_party_camera_frames[-1])
            image.save(f"{path}/{self.scene}_third_party{image_name}.png", format="PNG")
            kargs.pop("third_party_camera_frames")
        
        if "no_agent_view" not in kargs.keys():
            image = Image.fromarray(self.controller.last_event.frame)
            image.save(f"{path}/{self.scene}{image_name}.png", format="PNG")

        return f"{path}/{self.scene}{image_name}.png"

    def arm_reset(self):
        try:
            self.controller.step(
                action="MoveArm",
                position=dict(x=0, y=0, z=-1),
                coordinateSpace="armBase",
                restrictMovement=False,
                speed=1,
                returnToStart=True,
                fixedDeltaTime=0.02
            )
        except Exception as e:
            print(e)

    def adjust_agent_fieldOfView(self, fieldOfView):
        self.backup()
        self.controller.reset(self.scene, fieldOfView=fieldOfView)
        self.recover()

    def backup(self):
        angent_position = self.get_agent_position()
        angent_rotation = self.get_agent_rotation()
        agent_horizon = self.get_agent_horizon()
        self.agent_state.append([angent_position, angent_rotation, agent_horizon])
        for item in self.eventobject.get_objects(self.controller.last_event)[0]:
            self.object_state[item["name"]] = item

    def recover(self):
        angent_position = self.agent_state[-1][0]
        angent_rotation = self.agent_state[-1][1]
        agent_horizon = self.agent_state[-1][2]
        self.action.action_mapping["teleport"](self.controller, position=angent_position, rotation=angent_rotation, horizon=agent_horizon)
        self.update_event()

    def compute_position(self, item):
        target_position = None
        target_rotation = None
        event = self.controller.step(dict(action='GetInteractablePoses', objectId=item['objectId']))
        reachable_positions = event.metadata['actionReturn']
        if len(reachable_positions) == 0:
            print("No reachable positions found.")
            return target_position, target_rotation
        front_positions=[]
        side_positions=[]
        
        for position in reachable_positions:
            if round(abs(position['rotation'] - item['rotation']['y'])) == 180:
                front_positions.append(position)
            elif round(abs(position['rotation'] - item['rotation']['y'])) == 90:
                side_positions.append(position)
        
        if len(front_positions) > 0:
            max_distance = 0
            for position in front_positions:
                distance = math.sqrt((position['x'] - item['position']['x'])**2 + (position['z'] - item['position']['z'])**2)
                if distance > max_distance:
                    max_distance = distance
                    target_position = position
        
        if target_position is None and len(side_positions) > 0:
            max_distance = 0
            for position in side_positions:
                distance = math.sqrt((position['x'] - item['position']['x'])**2 + (position['z'] - item['position']['z'])**2)
                if distance > max_distance:
                    max_distance = distance
                    target_position = position
        
        if target_position is None:
            max_distance = 0
            for position in reachable_positions:
                distance = math.sqrt((position['x'] - item['position']['x'])**2 + (position['z'] - item['position']['z'])**2)
                if distance > max_distance:
                    max_distance = distance
                    target_position = position

        return target_position, dict(x=0, y=target_position['rotation'], z=0)

    def compute_position_1(self, item, reachable_positions):
        target_position = None
        target_rotation = None
        min_distance = float('inf')
        for position in reachable_positions:
            distance = math.sqrt((position['x'] - item['position']['x'])**2 + (position['z'] - item['position']['z'])**2)
            if distance < min_distance:
                min_distance = distance
                target_position = position
        rotation = target_position['rotation'] if "rotation" in target_position.keys() else 0
        return target_position, dict(x=0, y=rotation, z=0)

    def compute_position_8(self, item, pre_target_positions):
        target_position = None
        target_rotation = None
        event = self.controller.step(dict(action='GetInteractablePoses', objectId=item['objectId']))
        # event = self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = event.metadata['actionReturn']
        reachable_positions = [position for position in reachable_positions if math.sqrt((position['x'] - item['position']['x'])**2 + (position['z'] - item['position']['z'])**2) <= 1.5]
        if len(reachable_positions) == 0:
            print("No reachable positions found.")
            return target_position, target_rotation
        if pre_target_positions != []:
            reachable_positions = [position for position in reachable_positions \
                               if position not in pre_target_positions]
        if self.eventobject.get_item_volume(self.controller.last_event, item['name']) <= 0.1 and self.eventobject.get_item_surface_area(self.controller.last_event, item['name']) <= 1:
            target_position, target_rotation = self.compute_position_1(item, reachable_positions)
            return target_position, target_rotation
        # Possible angles to choose from
        angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        item_rotation = min(angles, key=lambda angle: abs(angle - round(item["rotation"]['y'])))
        item_rotation = 0 if item_rotation == 360 else item_rotation
        target_position = None
        target_rotation = None
        candidate_positions = []

        if item_rotation == 180:
            target_rotation = dict(x=0, y=0, z=0)
            for tolerance in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if len(candidate_positions) == 0:
                    candidate_positions = [position for position in reachable_positions if abs(position['x'] - item['position']['x']) <= tolerance]
                else:
                    break
            front_positions = []
            back_positions = []         
            for position in candidate_positions:
                if position['z'] < item['position']['z']:
                    front_positions.append(position)
                elif position['z'] > item['position']['z']:
                    back_positions.append(position)

            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)
            
            if target_rotation is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=180, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 270:
            target_rotation = dict(x=0, y=90, z=0)
            for tolerance in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if len(candidate_positions) == 0:
                    candidate_positions = [position for position in reachable_positions if abs(position['z'] - item['position']['z']) <= tolerance]
                else:
                    break
            front_positions = []
            back_positions = []
            for position in candidate_positions:
                if position['x'] < item['position']['x']:
                    front_positions.append(position)
                elif position['x'] > item['position']['x']:
                    back_positions.append(position)

            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)
            
            if target_position is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=270, z=0)
                target_position = self.compute_closest_positions(item, back_positions)
            
        elif item_rotation == 0:
            target_rotation = dict(x=0, y=180, z=0)
            for tolerance in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if len(candidate_positions) == 0:
                    candidate_positions = [position for position in reachable_positions if abs(position['z'] - item['position']['z']) <= tolerance]
                else:
                    break

            front_positions = []
            back_positions = []
            for position in candidate_positions:
                if position['z'] > item['position']['z']:
                    front_positions.append(position)
                elif position['z'] < item['position']['z']:
                    back_positions.append(position)
            
            target_position_front=None
            if len(front_positions) > 0:
                target_position_front = self.compute_closest_positions(item, front_positions)
            
            target_position_back=None
            if len(back_positions) > 0:
                target_rotation = dict(x=0, y=0, z=0)
                target_position_back = self.compute_closest_positions(item, back_positions)

            if target_position_front is not None and target_position_back is not None:
                distance_front = math.sqrt((target_position_front['x'] - item['position']['x'])**2 + (target_position_front['z'] - item['position']['z'])**2)
                distance_back = math.sqrt((target_position_back['x'] - item['position']['x'])**2 + (target_position_back['z'] - item['position']['z'])**2)
                if distance_front < distance_back:
                    target_position = target_position_front
                else:
                    target_position = target_position_back
            elif target_position_front is not None:
                target_position = target_position_front
            elif target_position_back is not None:
                target_position = target_position_back

        elif item_rotation == 90:
            target_rotation = dict(x=0, y=270, z=0)
            for tolerance in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if len(candidate_positions) == 0:
                    candidate_positions = [position for position in reachable_positions if abs(position['z'] - item['position']['z']) <= tolerance]
                else:
                    break
            front_positions = []
            back_positions = []
            for position in candidate_positions:
                if position['x'] > item['position']['x']:
                    front_positions.append(position)
                elif position['x'] < item['position']['x']:
                    back_positions.append(position)
            
            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)
            
            elif len(back_positions) > 0:
                target_rotation = dict(x=0, y=90, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 45:
            target_rotation = dict(x=0, y=225, z=0)
            front_positions = []
            back_positions = []
            for position in reachable_positions:
                if position['x'] > item['position']['x'] and position['z'] > item['position']['z']:
                    front_positions.append(position)
                elif position['x'] < item['position']['x'] and position['z'] < item['position']['z']:
                    back_positions.append(position)
            
            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)

            if target_position is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=45, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 135:
            target_rotation = dict(x=0, y=315, z=0)
            front_positions = []
            back_positions = []
            front_side_positions = []
            back_side_positions = []
            for position in reachable_positions:
                if position['x'] > item['position']['x'] and position['z'] < item['position']['z']:
                    front_positions.append(position)
                elif position['x'] < item['position']['x'] and position['z'] > item['position']['z']:
                    back_positions.append(position)
                elif position['x'] > item['position']['x'] and position['z'] > item['position']['z']:
                    front_side_positions.append(position)
                elif position['x'] < item['position']['x'] and position['z'] < item['position']['z']:
                    back_side_positions.append(position)
            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)
            if target_position is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=135, z=0)
                target_position = self.compute_closest_positions(item, back_positions)
            if target_position is None and len(front_side_positions) > 0:
                target_rotation = dict(x=0, y=225, z=0)
                target_position = self.compute_closest_positions(item, front_side_positions)
            if target_position is None and len(back_side_positions) > 0:
                target_rotation = dict(x=0, y=45, z=0)
                target_position = self.compute_closest_positions(item, back_side_positions)

        elif item_rotation == 225:
            target_rotation = dict(x=0, y=45, z=0)
            front_positions = []
            back_positions = []
            for position in reachable_positions:
                if position['x'] < item['position']['x'] and position['z'] < item['position']['z']:
                    front_positions.append(position)
                elif position['x'] > item['position']['x'] and position['z'] > item['position']['z']:
                    back_positions.append(position)

            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)
            
            if target_position is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=225, z=0)
                target_position = self.compute_closest_positions(item, back_positions)
                
        elif item_rotation == 315:
            target_rotation = dict(x=0, y=135, z=0)
            front_positions = []
            back_positions = []
            for position in reachable_positions:
                if position['x'] < item['position']['x'] and position['z'] > item['position']['z']:
                    front_positions.append(position)
                elif position['x'] > item['position']['x'] and position['z'] < item['position']['z']:
                    back_positions.append(position)
            
            if len(front_positions) > 0:
                target_position = self.compute_closest_positions(item, front_positions)

            if target_position is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=315, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        if target_position is None:
            target_position, target_rotation = self.compute_position_1(item, reachable_positions)
        return target_position, target_rotation
    
    def compute_closest_positions_xxx(self, item, candidate_positions, gap=0.1):
        item_position = item["position"]
        item_volume = self.eventobject.get_item_volume(self.controller.last_event, item['name'])
        item_surface_area = self.eventobject.get_item_surface_area(self.controller.last_event, item['name'])
        target_position = None
        A = 1
        B = -math.tan(math.radians(item['rotation']['y']))
        C = -item_position['x'] - B * item_position['z']
        if item_surface_area > 0.5 or item_volume>0.5:
            min_dinstance = float('inf')
            closest_points = []
            for position in candidate_positions:
                x0 = position['x']
                z0 = position['z']
                numerator = abs(A * z0 + B * x0 + C)
                denominator = math.sqrt(A**2 + B**2)
                distance =  numerator / denominator
                if distance <= min_dinstance + gap:
                    if distance < min_dinstance:
                        min_dinstance = distance
                    closest_points.append(position)
            
            closest_points = sorted(closest_points, key=lambda position: math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2))
            if item_surface_area <= 1:
                target_position = closest_points[0] if len(closest_points) != 0 else None
            elif item_surface_area <= 2:
                closest_points = [position for position in candidate_positions
                                if 0.5<=math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1]
                target_position = closest_points[0] if len(closest_points) != 0 else None
            else:
                closest_points = [position for position in candidate_positions
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1]
                target_position = closest_points[len(closest_points)//2] if len(closest_points) != 0 else None

            # else:
            #     closest_points = [position for position in candidate_positions 
            #                     if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1.5]
            #     target_position = closest_points[-1] if len(closest_points) != 0 else None
            return target_position
        else:
            closest_points = []
            
            if item_volume <= 0.2 and item_surface_area <=0.5:
                closest_points = [position for position in candidate_positions 
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 0.5]
            
            elif item_volume <= 1 and item_surface_area <= 1:
                closest_points = [position for position in candidate_positions 
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1]
            
            elif item_volume <= 2 and item_surface_area <= 2:
                closest_points = [position for position in candidate_positions 
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1.5]
            else:
                closest_points = [position for position in candidate_positions 
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 2]

            # closest_points = sorted(closest_points, key=lambda position: math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2))
            # closest_points = closest_points[len(closest_points)//2:]
            
            min_dinstance = float('inf')
            for position in closest_points:
                x0 = position['x']
                z0 = position['z']
                numerator = abs(A * z0 + B * x0 + C)
                denominator = math.sqrt(A**2 + B**2)
                distance =  numerator / denominator
                if distance < min_dinstance:
                    min_dinstance = distance
                    target_position = position
            
            return target_position

    def compute_closest_positions(self, item, candidate_positions, gap=0.1):
        item_position = item["position"]
        item_volume = self.eventobject.get_item_volume(self.controller.last_event, item['name'])
        item_surface_area = self.eventobject.get_item_surface_area(self.controller.last_event, item['name'])
        A = 1
        B = -math.tan(math.radians(item['rotation']['y']))
        C = -item_position['x'] - B * item_position['z']
        
        min_dinstance = float('inf')
        closest_points = []
        for position in candidate_positions:
            x0 = position['x']
            z0 = position['z']
            numerator = abs(A * z0 + B * x0 + C)
            denominator = math.sqrt(A**2 + B**2)
            distance =  numerator / denominator
            if distance <= min_dinstance + gap:
                if distance < min_dinstance:
                    min_dinstance = distance
                closest_points.append(position)

        if item_volume <= 0.2 and item_surface_area <=0.5:
            min_distance = float('inf')
            for position in closest_points:
                distance = math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2)
                if distance < min_distance:
                    min_distance = distance
                    target_position = position
            return target_position
        
        elif item_volume <= 1 and item_surface_area <= 1:
            closest_points = [position for position in closest_points
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1]
            closest_points = sorted(closest_points, key=lambda position: math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2))
            
            return closest_points[len(closest_points)//2] if len(closest_points) != 0 else None
        else:
            max_distance = 0
            target_position = None
            closest_points = [position for position in closest_points
                                if math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2) <= 1]
            for position in closest_points:
                distance = math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2)
                if distance > max_distance:
                    max_distance = distance
                    target_position = position
        
        return target_position
    
    def compute_position_(self, item):
        target_position = None
        target_rotation = None
        event = self.controller.step(dict(action='GetInteractablePoses', objectId=item['objectId']))
        # event = self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = event.metadata['actionReturn']
        if len(reachable_positions) == 0:
            print("No reachable positions found.")
            return target_position, target_rotation
        
        item_position = item["position"]
        
        # Possible angles to choose from
        angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        item_rotation = min(angles, key=lambda angle: abs(angle - round(item["rotation"]['y'])))
        item_rotation = 0 if item_rotation == 360 else item_rotation
        target_position = None
        target_rotation = None
        # min_distance_z = float('inf')
        # min_distance_x = float('inf')
        candidate_positions = []
        max_distance_z = 0
        max_distance_x = 0
        if item_rotation == 180:
            target_rotation = dict(x=0, y=0, z=0)
            for position in reachable_positions:
                if abs(position['x'] - item['position']['x'])<=0.05: # or abs(position['x'] - item['position']['x']) < min_distance_x) and (position['z'] < item['position']['z']
                    candidate_positions.append(position)
                        
            for position in candidate_positions:
                if position['z'] < item['position']['z'] and max_distance_z < abs(position['z'] - item['position']['z']):
                    target_position = position
                    max_distance_z = abs(position['z'] - item['position']['z'])
            
            if target_position is None:
                for position in candidate_positions:
                    if position['z'] > item['position']['z'] and max_distance_z < abs(position['z'] - item['position']['z']):
                        target_position = position
                        max_distance_z = abs(position['z'] - item['position']['z'])        
                target_rotation = dict(x=0, y=180, z=0)
            
        elif item_rotation == 270:
            target_rotation = dict(x=0, y=90, z=0)
            for position in reachable_positions:
                if abs(position['z'] - item['position']['z'])<=0.05:
                    candidate_positions.append(position)

            for position in candidate_positions:
                if position['x'] < item['position']['x'] and max_distance_x < abs(position['x'] - item['position']['x']):
                    target_position = position
                    max_distance_x = abs(position['x'] - item['position']['x'])

            if target_position is None:
                for position in candidate_positions:
                    if position['x'] > item['position']['x'] and max_distance_x < abs(position['x'] - item['position']['x']):
                        target_position = position
                        max_distance_x = abs(position['x'] - item['position']['x'])
                target_rotation = dict(x=0, y=270, z=0)
                
        elif item_rotation == 0:
            target_rotation = dict(x=0, y=180, z=0)
            for position in reachable_positions:
                if abs(position['x'] - item['position']['x'])<=0.05:
                    candidate_positions.append(position)

            for position in candidate_positions:
                if position['z'] > item['position']['z'] and max_distance_z < abs(position['z'] - item['position']['z']):
                    target_position = position
                    max_distance_z = abs(position['z'] - item['position']['z'])
            
            if target_position is None:
                for position in candidate_positions:
                    if position['z'] < item['position']['z'] and max_distance_z < abs(position['z'] - item['position']['z']):
                        target_position = position
                        max_distance_z = abs(position['z'] - item['position']['z'])
                target_rotation = dict(x=0, y=0, z=0)
        
        elif item_rotation == 90:
            target_rotation = dict(x=0, y=270, z=0)
            for position in reachable_positions:
                if abs(position['z'] - item['position']['z'])<=0.05:
                    candidate_positions.append(position)
            
            for position in candidate_positions:
                if position['x'] > item['position']['x'] and max_distance_x < abs(position['x'] - item['position']['x']):
                    target_position = position
                    max_distance_x = abs(position['x'] - item['position']['x'])

            if target_position is None:
                for position in candidate_positions:
                    if position['x'] < item['position']['x'] and max_distance_x < abs(position['x'] - item['position']['x']):
                        target_position = position
                        max_distance_x = abs(position['x'] - item['position']['x'])
                target_rotation = dict(x=0, y=90, z=0)        

        elif item_rotation == 45:
            target_rotation = dict(x=0, y=225, z=0)
            front_positions = []
            back_positions = []
            for position in reachable_positions:
                if position['x'] > item['position']['x'] and position['z'] > item['position']['z']:
                    front_positions.append(position)
                elif position['x'] < item['position']['x'] and position['z'] < item['position']['z']:
                    back_positions.append(position)
            
            if len(front_positions) > 0:
                A = 1
                B = -math.tan(math.radians(item['rotation']['y']))
                C = -item_position['x'] - B * item_position['z']
                
                min_dinstance = float('inf')
                closest_points = []
                for position in front_positions:
                    x0 = position['x']
                    z0 = position['z']
                    numerator = abs(A * z0 + B * x0 + C)
                    denominator = math.sqrt(A**2 + B**2)
                    distance =  numerator / denominator
                    if distance <= min_dinstance + 0.1:
                        if distance < min_dinstance:
                            min_dinstance = distance
                        closest_points.append(position)

                max_distance = 0
                for position in closest_points:
                    distance = math.sqrt((position['x'] - item_position['x'])**2 + (position['z'] - item_position['z'])**2)
                    if distance > max_distance:
                        max_distance = distance
                        target_position = position

            if target_position is None and len(back_positions) > 0:
                target_rotation = dict(x=0, y=45, z=0)
                max_distance = 0
                for position in back_positions:
                    distance = math.sqrt((position['x'] - item['position']['x'])**2 + (position['z'] - item['position']['z'])**2)
                    if distance > max_distance:
                        max_distance = distance
                        target_position = position

        elif item_rotation == 135:
            target_rotation = dict(x=0, y=315, z=0)
            front_positions = []
            back_positions = []
            for position in reachable_positions:
                if position['x'] < item['position']['x'] and position['z'] > item['position']['z']:
                    front_positions.append(position)
                elif position['x'] > item['position']['x'] and position['z'] < item['position']['z']:
                    back_positions.append(position)


        elif item_rotation >0 and item_rotation < 90:
            for position in reachable_positions:
                if position['x'] > item['position']['x'] and position['z'] > item['position']['z']:
                    candidate_positions.append(position)
            
            min_rotation_gap = float('inf')
            for position in candidate_positions:
                dz = position['z'] - item_position['z']
                dx = position['x'] - item_position['x']
                angle_radians = math.atan2(dz, dx)
                angle_degrees = math.degrees(angle_radians)+180
                if abs(angle_degrees - item_rotation) < min_rotation_gap:
                    min_rotation_gap = abs(angle_degrees - item_rotation)
                    target_position = position

            target_rotation = dict(x=0, y=angle_degrees, z=0)#...............

        elif item_rotation >90 and item_rotation < 180:

            for position in reachable_positions:
                if position['x'] > item['position']['x'] and position['z'] < item['position']['z']:
                    candidate_positions.append(position)
            
            for position in candidate_positions:
                dz = position['z'] - item_position['z']
                dx = position['x'] - item_position['x']
                angle_radians = math.atan2(dz, dx)
                angle_degrees = math.degrees(angle_radians)+360
                if abs(angle_degrees - item_rotation) < min_rotation_gap:
                    min_rotation_gap = abs(angle_degrees - item_rotation)
                    target_position = position

            target_rotation = dict(x=0, y=angle_degrees, z=0)#...............

        elif item_rotation >180 and item_rotation < 270:

            for position in reachable_positions:
                if position['x'] < item['position']['x'] and position['z'] < item['position']['z']:
                    candidate_positions.append(position)

            for position in candidate_positions:
                dz = position['z'] - item_position['z']
                dx = position['x'] - item_position['x']
                angle_radians = math.atan2(dz, dx)
                angle_degrees = math.degrees(angle_radians)
                if abs(angle_degrees - item_rotation) < min_rotation_gap:
                    min_rotation_gap = abs(angle_degrees - item_rotation)
                    target_position = position
            
            target_rotation = dict(x=angle_degrees, y=180, z=0)#...............
        
        elif item_rotation >270 and item_rotation < 360:
            
            for position in reachable_positions:
                if position['x'] < item['position']['x'] and position['z'] > item['position']['z']:
                    candidate_positions.append(position)

            for position in candidate_positions:
                dz = position['z'] - item_position['z']
                dx = position['x'] - item_position['x']
                angle_radians = math.atan2(dz, dx)
                angle_degrees = math.degrees(angle_radians)+180
                if abs(angle_degrees - item_rotation) < min_rotation_gap:
                    min_rotation_gap = abs(angle_degrees - item_rotation)
                    target_position = position

            target_rotation = dict(x=0, y=angle_degrees, z=0)#...............
        # target_position = reachable_positions[0] if target_position is None else target_position
        # target_rotation = dict(x=0, y=target_position['rotation'], z=0)
        if target_position is None:

            A = 1
            B = -math.tan(math.radians(item['rotation']['y']))
            C = -item_position['x'] - B * item_position['z']
            
            min_dinstance = float('inf')
            closest_point = None
            for position in reachable_positions:
                x0 = position['x']
                z0 = position['z']
                numerator = abs(A * z0 + B * x0 + C)
                denominator = math.sqrt(A**2 + B**2)
                distance =  numerator / denominator
                if distance <= min_dinstance:
                    min_dinstance = distance
                    closest_point = position

            dz = closest_point['z'] - item_position['z']
            dx = closest_point['x'] - item_position['x']
            angle_radians = math.atan2(dz, dx)
            angle_degrees = math.degrees(angle_radians)
            angle_degrees = angle_degrees + 360 if angle_degrees < 0 else angle_degrees

            rotation = dict(x=0, y=closest_point['rotation'], z=0) if 'rotation' in closest_point.keys() else dict(x=0, y=angle_degrees, z=0)
            # ##################################################################################

            return closest_point, rotation
        
        
        return target_position, target_rotation

    def calculate_best_view_angles(self, item):
        import numpy as np
        camera_position = self.get_camera_position()

        look_vector = np.array([
            item["axisAlignedBoundingBox"]["center"]["x"] - camera_position["x"],
            item["axisAlignedBoundingBox"]["center"]["y"] - camera_position["y"],
            item["axisAlignedBoundingBox"]["center"]["z"] - camera_position["z"]
        ])

        norm = np.linalg.norm(look_vector)
        look_vector = look_vector / norm if norm != 0 else look_vector

        yaw = np.arctan2(look_vector[0], look_vector[2])
        pitch = np.arcsin(look_vector[1])

        return np.degrees(yaw), np.degrees(pitch)


    def adjust_view(self, item):
        if item["objectId"] not in self.objid2position:

            yaw_degrees, pitch_degrees = self.calculate_best_view_angles(item)
            
            camera_yaw = -self.get_camera_rotation()

            if int(pitch_degrees) > 0:
                target_yaw = min(30, int(pitch_degrees)) # look up
            else:
                target_yaw = max(-60, int(pitch_degrees)) # look down

            # -51 -60
            if target_yaw - camera_yaw > 0: # look up 80
                self.action.action_mapping["look_up"](self.controller, target_yaw - camera_yaw)
            # 0, 30
            elif target_yaw - camera_yaw <0:
                self.action.action_mapping["look_down"](self.controller, camera_yaw - target_yaw)

            self.update_event()

            if int(yaw_degrees) < 0:
                yaw_degrees = 360 + yaw_degrees
            
            angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
            yaw_rotation = min(angles, key=lambda angle: abs(angle - round(yaw_degrees)))
            yaw_rotation = 0 if yaw_rotation == 360 else yaw_rotation
            # if item['name'] not in self.eventobject.get_visible_objects()[0]:
            self.action.action_mapping["rotate_left"](self.controller, self.get_agent_rotation()['y']-yaw_rotation)
            self.update_event()

    def adjust_height(self, item):
        if item["objectId"] in self.objid2position:
            agent_isstanding = self.objid2position[item["objectId"]]["agent_isstanding"]
            if agent_isstanding:
                if (self.controller.last_event.metadata["agent"]["isStanding"]==False):
                    self.action.action_mapping["stand"](self.controller)
            else:
                if (self.controller.last_event.metadata["agent"]["isStanding"]==True):
                    self.action.action_mapping["crouch"](self.controller)
        else:
            if self.get_agent_position()['y'] > item['axisAlignedBoundingBox']['cornerPoints'][0][1] + 0.44:
                self.action.action_mapping["crouch"](self.controller)
            else:
                self.action.action_mapping["stand"](self.controller)
            
            self.update_event()
            if item['name'] not in self.eventobject.get_visible_objects(self.controller.last_event)[0]:
                self.action.action_mapping["stand"](self.controller)
                self.update_event()

    def get_edge_init_view(self):
        scene_bounds2 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][2]
        scene_bounds3 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][3]
        scene_bounds6 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][6]
        scene_bounds7 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][7]
        edge23 = math.sqrt((scene_bounds2[0]-scene_bounds3[0])**2 + (scene_bounds2[2]-scene_bounds3[2])**2)
        edge26 = math.sqrt((scene_bounds2[0]-scene_bounds6[0])**2 + (scene_bounds2[2]-scene_bounds6[2])**2)
        edge37 = math.sqrt((scene_bounds3[0]-scene_bounds7[0])**2 + (scene_bounds3[2]-scene_bounds7[2])**2)
        edge67 = math.sqrt((scene_bounds6[0]-scene_bounds7[0])**2 + (scene_bounds6[2]-scene_bounds7[2])**2)
        
        min_edge = max(edge23, edge26, edge37, edge67)
        if min_edge == edge23:
            center = [(scene_bounds2[0]+scene_bounds3[0])/2, (scene_bounds2[1]+scene_bounds3[1])/2, (scene_bounds2[2]+scene_bounds3[2])/2]
            # 180-360
            target_rotation = dict(x=0, y=225, z=0)
            # target_position = dict(x=0, y=315, z=0)
        elif min_edge == edge26:
            center = [(scene_bounds2[0]+scene_bounds6[0])/2, (scene_bounds2[1]+scene_bounds6[1])/2, (scene_bounds2[2]+scene_bounds6[2])/2]
            # 90-270
            target_position = dict(x=0, y=135, z=0)
            # target_rotation = dict(x=0, y=225, z=0)
        elif min_edge == edge37:
            center = [(scene_bounds3[0]+scene_bounds7[0])/2, (scene_bounds3[1]+scene_bounds7[1])/2, (scene_bounds3[2]+scene_bounds7[2])/2]
            # 0-90；270-360
            target_rotation = dict(x=0, y=45, z=0)
            # target_position = dict(x=0, y=315, z=0)
        else:
            # 0-90-180
            target_rotation = dict(x=0, y=45, z=0)
            # target_position = dict(x=0, y=135, z=0)
            center = [(scene_bounds6[0]+scene_bounds7[0])/2, (scene_bounds6[1]+scene_bounds7[1])/2, (scene_bounds6[2]+scene_bounds7[2])/2]
        
        event = self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = event.metadata['actionReturn']
        min_distance = float("inf")
        for position in reachable_positions:
            distance = math.sqrt((position['x']-center[0])**2 + (position['z']-center[2])**2)
            if distance < min_distance:
                min_distance = distance
                target_position = position

        
        self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=0)
        
        self.save_frame({"action": "init_view1"}, prefix_save_path="./data/init_scene_image")
        self.action.action_mapping["rotate_right"](self.controller, 90)
        self.save_frame({"action": "init_view2"}, prefix_save_path="./data/init_scene_image")
        pass
    