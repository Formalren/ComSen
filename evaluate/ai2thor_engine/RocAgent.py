import math
try:
    from utils import *
except Exception as e:
    print(e)
try:
    from .utils import add_text_to_image, add_border, EventObject
except Exception as e:
    print(e)

from .baseAgent import BaseAgent
from tqdm import tqdm
import numpy as np
import cv2, json

OBSERVE_API_MAX_DIM = 2048


def resize_for_api_limit(image, max_dim=OBSERVE_API_MAX_DIM):
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dim:
        return image
    scale = max_dim / float(largest)
    resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)


class RocAgent(BaseAgent):
    STATE_OBSERVATION = "observation"
    STATE_PLANNING = "planning"
    STATE_THINKING = "thinking"
    STATE_REFLECTION = "reflection"
    STATE_DECISION_MAKING_STATE = "decision_making"
    STATE_VERIFICATION = "verification"
    STATE_END = "end"
    def __init__(self, controller, save_path="./data/", scene="FloorPlan203", 
                 visibilityDistance=1.5, gridSize=0.25, fieldOfView=90, target_objects=[], related_objects=[], navigable_objects=[], taskid=0,platform_type="GPU"):
        super().__init__(controller, scene, visibilityDistance, gridSize, fieldOfView,platform_type)
        self.env, self.executor, self.monitor, self.planner = self.build_agent()
        self.pre_navigate_location=""
        self.agent_state = []
        self.object_state = {}
        self.target_objects = []
        self.navigale_objects = {}
        self.state = ""
        self.result_dir = f"{save_path}"
        self.navigable_objects = {}
        self.legal_interactions = {}
        self.current_container = None
        self.objecttype2object={}
        self.action_space = {
            "init": self.init_agent_corner,
            "navigate to": self.navigate,
            "pickup": self.pick_up,
            "put": self.put_in,
            "put in":self.put_in,   # for MODE=API
            "toggle": self.toggle,
            "open": self.open,
            "close": self.close,
            "break": self.break_obj,
            "slice": self.slice_obj,
            "observe": self.observe,
            "move forward": self.move_forward,
            "end": "end",
        }
        self.related_objects=related_objects
        self.target_item_type2obj_id = {}
        for target_obj in target_objects:
            key = target_obj.split("|")[0].replace("*", "")
            if key not in self.target_item_type2obj_id:
                self.target_item_type2obj_id[key] = []
            self.target_item_type2obj_id[key].append(target_obj)
        
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectType'] not in self.objecttype2object:
                self.objecttype2object[obj['objectType']]=[]
            self.objecttype2object[obj['objectType']].append(obj)
        
        for navigable_obj in navigable_objects:
            if navigable_obj not in self.navigable_objects:
                self.navigable_objects[navigable_obj] = 0
            self.navigable_objects[navigable_obj] += 1
        self.taskid = str(taskid)
        self.objid2position={}
        with open("./data/agent_positions.json") as f:
            custom_position_data = json.load(f)
        for taskid in custom_position_data:
            temp_data = custom_position_data[taskid]
            for objid in temp_data:
                if objid != "scene" and objid != "tasktype" and objid != "taskname":
                    self.objid2position[objid] = temp_data[objid]

        # if self.taskid in custom_position_data:
        #     self.objid2position = custom_position_data[self.taskid]
        # self.init_agent_corner()
        
    def build_agent(self):
        return None, None, None, None


    def init_agent_corner(self):
        scene_bounds2 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][2]
        scene_bounds3 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][3]
        scene_bounds6 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][6]
        scene_bounds7 = self.controller.last_event.metadata['sceneBounds']['cornerPoints'][7]

        event = self.controller.step(dict(action='GetReachablePositions'))
        reachable_positions = event.metadata['actionReturn']
        pre_target_positions = []
        min_distance = float("inf")
        for i, scene_bounds in enumerate([scene_bounds2, scene_bounds3, scene_bounds6, scene_bounds7]):
            for position in reachable_positions:
                distance = math.sqrt((position['x']-scene_bounds[0])**2 + (position['z']-scene_bounds[2])**2)
                if distance < min_distance:
                    min_distance = distance
                    target_position = position
                    index = i
        if index == 0:
            # 180, 270
            target_rotation = dict(x=0, y=225, z=0)
        elif index == 1:
            # 270, 360
            target_rotation = dict(x=0, y=315, z=0)
        elif index == 2:
            # 90,180
            target_rotation = dict(x=0, y=135, z=0)
        else:
            # 0,90
            target_rotation = dict(x=0, y=45, z=0)
        
        while True:
            event = self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=0)
            self.update_event()
            if event.metadata['lastActionSuccess']:
                break
            else:
                pre_target_positions.append(target_position)
                event = self.controller.step(dict(action='GetReachablePositions'))
                reachable_positions = event.metadata['actionReturn']
                
                min_distance = float("inf")
                for i, scene_bounds in enumerate([scene_bounds2, scene_bounds3, scene_bounds6, scene_bounds7]):
                    for position in reachable_positions:
                        if position in pre_target_positions:
                            continue
                        distance = math.sqrt((position['x']-scene_bounds[0])**2 + (position['z']-scene_bounds[2])**2)
                        if distance < min_distance:
                            min_distance = distance
                            target_position = position
                            index = i
                if index == 0:
                    # 180, 270
                    target_rotation = dict(x=0, y=225, z=0)
                elif index == 1:
                    # 270, 360
                    target_rotation = dict(x=0, y=315, z=0)
                elif index == 2:
                    # 90,180
                    target_rotation = dict(x=0, y=135, z=0)
                else:
                    # 0,90
                    target_rotation = dict(x=0, y=45, z=0)
                print("Teleport failed, retrying...")
        self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=0)
        self.update_event()
        # self.save_frame({"action": "init_agent_view"}, prefix_save_path="./data/init_scene_image")
        # self.action.action_mapping["rotate_right"](self.controller, 30)
        # self.update_legal_location()
        # self.save_frame({"action": "init_view2"}, prefix_save_path="./data/init_scene_image")
        image_fp, legal_navigations, legal_interactions = None, None, None
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "init",},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def navigate(self, itemtype):
        image_fp, legal_navigations, legal_interactions = None, None, None
        # tolerate newly created object types (e.g., *Sliced after slice)
        def _find_object_by_type(obj_type):
            base = obj_type.replace("*", "")
            for obj in self.controller.last_event.metadata['objects']:
                if obj['objectType'] == obj_type or obj['objectType'].startswith(base):
                    return obj
            return None
        # print("target_item_type2obj_id",self.target_item_type2obj_id)
        if itemtype in self.target_item_type2obj_id:
            if self.taskid=="1900" or self.taskid=="1850":
                if self.controller.last_event.metadata["inventoryObjects"] == []:
                    obj_id = self.target_item_type2obj_id[itemtype][0]
                else:
                    obj_id = self.target_item_type2obj_id[itemtype][1]
            else:
                obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
            if item is None:
                item = _find_object_by_type(itemtype)
                if item:
                    self.objecttype2object.setdefault(itemtype, []).insert(0, item)
                else:
                    return image_fp, self.get_legal_navigations(), self.get_legal_interactions()
        else:
            if itemtype not in self.objecttype2object:
                obj = _find_object_by_type(itemtype)
                if obj:
                    self.objecttype2object[itemtype] = [obj]
                else:
                    return image_fp, self.get_legal_navigations(), self.get_legal_interactions()
            item = self.objecttype2object[itemtype][0]

        if item is None:
            item = _find_object_by_type(itemtype)
            if item:
                self.objecttype2object[itemtype] = [item]
            else:
                return image_fp, self.get_legal_navigations(), self.get_legal_interactions()
        
        navigate_obj_type=item["objectType"]
        
        if item.get("receptacle", False) and (not item["openable"]):
            for related_object in self.related_objects:
                if related_object in item['receptacleObjectIds']:
                    item = self.eventobject.get_object_by_id(self.controller.last_event, related_object)
                    break
        
        # while(item['name'] == self.pre_navigate_location and len(self.objecttype2object[item['objectType']])>1):
        #     item = random.choice(self.objecttype2object[item['objectType']])
        # self.pre_navigate_location = item['name']
        if item["objectId"] in self.objid2position:
            target_position = self.objid2position[item["objectId"]]["agent_teleport_position"]
            target_rotation = self.objid2position[item["objectId"]]["agent_rotation"]
            horizon = self.objid2position[item["objectId"]]["agent_cameraHorizon"]
            print("", self.objid2position)
        else:
            target_position, target_rotation = self.compute_position_8(item, pre_target_positions=[])
            horizon = 60
        # self.arm_reset()
        if target_position is None:
            print("teleport failed, no reachable positions")
            return image_fp, legal_navigations, legal_interactions
        event = self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation, horizon=horizon)
        pre_target_positions = []
        index = 0
        while not event.metadata['lastActionSuccess']:
            index += 1
            print(f"teleport failed, retrying...{index}")
            pre_target_positions.append(target_position)
            target_position, target_rotation = self.compute_position_8(item, pre_target_positions)
            event = self.action.action_mapping["teleport"](self.controller, position=target_position, rotation=target_rotation)
            self.update_event()
        
        if item["objectId"] not in self.objid2position:
            self.adjust_height(item)
            self.adjust_view(item)

        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "navigate",
                                    "item": navigate_obj_type},
                                    prefix_save_path=self.result_dir)
        
        
        if item.get("receptacle", False) and "receptacleObjectIds" in item and (item['receptacleObjectIds'] != [] or item['receptacleObjectIds'] is not None):
            self.current_container = item
        
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        
        # self.update_legal_location()
        return image_fp, legal_navigations, legal_interactions
    
    def observe(self):
        image_fp, legal_navigations, legal_interactions = [], None, None
        for i in range(3):
            self.action.action_mapping["rotate_left"](self.controller, 90)
            
            image_fp.append(self.save_frame({"step_count": str(self.step_count),
                                        "i": str(i),
                                        "action": "observe"},
                                        prefix_save_path=self.result_dir))
            legal_navigations = self.get_legal_navigations()

        for i in range(3):
            images = [cv2.imread(path) for path in image_fp]
            img1 = add_text_to_image(images[0], "left view", (10, images[0].shape[0] - 20))
            img2 = add_text_to_image(images[1], "back view", (10, images[1].shape[0] - 20))
            img3 = add_text_to_image(images[2], "right view", (10, images[2].shape[0] - 25))
            img2_with_border = add_border(img2, 5, (0, 0, 0))
            cv2.imwrite("outpath1.jpg", img1)
            img_h_concat = np.concatenate((img1, img2_with_border, img3), axis=1)
            img_h_concat = resize_for_api_limit(img_h_concat)
            print(img_h_concat.shape)
            
            if img1 is None:
                print("：！")
            else:
                cv2.imwrite("output.jpg", img1)
            
            output_path = self.save_frame({"step_count": str(self.step_count),
                                            # "i": str(i),
                                            "action": "observe"},
                                            prefix_save_path=self.result_dir)
            try:
                cv2.imwrite(output_path, img_h_concat)
                break
            except Exception as e:
                print("try_save_image")
                print(e)
        
        self.action.action_mapping["rotate_left"](self.controller, 90)
        legal_interactions = self.get_legal_interactions()
        
        return output_path, legal_navigations, legal_interactions
        
    def move_forward(self, distance=0.5):
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        
        self.action.action_mapping["move_ahead"](self.controller, distance)
        print("RocAgent",self.controller.last_event)
        if self.controller.last_event.metadata["errorMessage"]=="":
            image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
            legal_navigations = self.get_legal_navigations()
            legal_interactions = self.get_legal_interactions()
            return image_fp, legal_navigations, legal_interactions
        else:
  
            if self.related_objects:
                distance_right_list = []
                distance_left_list = []
                
                # move_r_or_l=random.choice(["move_right","move_left"])
                self.action.action_mapping["move_right"](self.controller, distance)
                print("RocAgent",self.controller.last_event)
                errorMessage1=self.controller.last_event.metadata["errorMessage"]
                agentxright=self.controller.last_event.metadata["agent"]["position"]["x"]
                agentzright=self.controller.last_event.metadata["agent"]["position"]["z"]  

                if errorMessage1=="":
                    self.action.action_mapping["move_left"](self.controller, distance)
                self.action.action_mapping["move_left"](self.controller, distance)
                print("RocAgent",self.controller.last_event)
                errorMessage2=self.controller.last_event.metadata["errorMessage"]
                agentxleft=self.controller.last_event.metadata["agent"]["position"]["x"]
                agentzleft=self.controller.last_event.metadata["agent"]["position"]["z"] 
                
                if errorMessage2=="":
                    self.action.action_mapping["move_right"](self.controller, distance)
                
                for obj_id in self.related_objects:
                    item = self.eventobject.get_object_by_id(self.controller.last_event,obj_id)
                    if item["visible"]==True:
                        itemx=item["position"]["x"]
                        itemz=item["position"]["z"]
                        
                 
                        distance_right = math.sqrt((agentxright - itemx) ** 2 + (agentzright - itemz) ** 2)
                        distance_right_list.append(distance_right)
                   
                        distance_left = math.sqrt((agentxleft - itemx) ** 2 + (agentzleft - itemz) ** 2)
                        distance_left_list.append(distance_left)
                   
                if errorMessage1=="" and errorMessage2=="" and distance_right_list and distance_left_list:# 

                    
                    min_distance_right = min(distance_right_list)
                    min_distance_left = min(distance_left_list)
                    
                    if min_distance_right < min_distance_left:
                        direction = "move_right"
                    else:
                        direction = "move_left"
                    
               
                    self.action.action_mapping[direction](self.controller, distance)
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                                "action": "move_forward"},
                                                prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions  
                    
                elif errorMessage1=="" or errorMessage2=="":  
                    if errorMessage1=="":
                        self.action.action_mapping["move_right"](self.controller, distance)
                        
                    elif errorMessage2=="":
                        self.action.action_mapping["move_left"](self.controller, distance)
                    
                    print("RocAgent",self.controller.last_event)                  
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                                "action": "move_forward"},
                                                prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions
                
                else:
                    self.action.action_mapping["move_back"](self.controller, distance) 
                    print("RocAgent",self.controller.last_event)
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions
                    
                    else:
                        self.action.action_mapping["rotate_right"](self.controller,degrees=90)
                        errorMessage_rotate_right=self.controller.last_event.metadata["errorMessage"]
                        self.action.action_mapping["move_ahead"](self.controller, distance)
                        print("RocAgent",self.controller.last_event)
                        if self.controller.last_event.metadata["errorMessage"]=="":
                            image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
                            legal_navigations = self.get_legal_navigations()
                            legal_interactions = self.get_legal_interactions()
                            return image_fp, legal_navigations, legal_interactions
                        else:
                            if errorMessage_rotate_right=="":
                                self.action.action_mapping["rotate_left"](self.controller,degrees=180)
                            self.action.action_mapping["move_ahead"](self.controller, distance)
                            print("RocAgent",self.controller.last_event)
                            if self.controller.last_event.metadata["errorMessage"]=="":
                                image_fp = self.save_frame({"step_count": str(self.step_count),
                                        "action": "move_forward"},
                                        prefix_save_path=self.result_dir)
                                legal_navigations = self.get_legal_navigations()
                                legal_interactions = self.get_legal_interactions()
                                return image_fp, legal_navigations, legal_interactions
                    
            else:
                self.action.action_mapping["move_right"](self.controller, distance)
                                    
                if self.controller.last_event.metadata["errorMessage"]=="":
                    image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                    legal_navigations = self.get_legal_navigations()
                    legal_interactions = self.get_legal_interactions()
                    return image_fp, legal_navigations, legal_interactions

                else:
                    self.action.action_mapping["move_left"](self.controller, distance)
                                        
                    if self.controller.last_event.metadata["errorMessage"]=="":
                        image_fp = self.save_frame({"step_count": str(self.step_count),
                                                "action": "move_forward"},
                                                prefix_save_path=self.result_dir)
                        legal_navigations = self.get_legal_navigations()
                        legal_interactions = self.get_legal_interactions()
                        return image_fp, legal_navigations, legal_interactions
                    else:
      
                        self.action.action_mapping["move_back"](self.controller, distance)
                        print("RocAgent",self.controller.last_event)
                        if self.controller.last_event.metadata["errorMessage"]=="":
                            image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                            legal_navigations = self.get_legal_navigations()
                            legal_interactions = self.get_legal_interactions()
                            return image_fp, legal_navigations, legal_interactions
                        
                        else:
                            self.action.action_mapping["rotate_right"](self.controller,degrees=90)
                            errorMessage_rotate_right=self.controller.last_event.metadata["errorMessage"]
                            self.action.action_mapping["move_ahead"](self.controller, distance)
                            print("RocAgent",self.controller.last_event)
                            if self.controller.last_event.metadata["errorMessage"]=="":
                                image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                                legal_navigations = self.get_legal_navigations()
                                legal_interactions = self.get_legal_interactions()
                                return image_fp, legal_navigations, legal_interactions
                            else:
                                if errorMessage_rotate_right=="":
                                    self.action.action_mapping["rotate_left"](self.controller,degrees=180)
                                self.action.action_mapping["move_ahead"](self.controller, distance)
                                # print("RocAgent",self.controller.last_event)
                                if self.controller.last_event.metadata["errorMessage"]=="":
                                    image_fp = self.save_frame({"step_count": str(self.step_count),
                                            "action": "move_forward"},
                                            prefix_save_path=self.result_dir)
                                    legal_navigations = self.get_legal_navigations()
                                    legal_interactions = self.get_legal_interactions()
                                    return image_fp, legal_navigations, legal_interactions
                    
        # print("RocAgent",self.controller.last_event)
        return image_fp, legal_navigations, legal_interactions

    def pick_up(self, itemtype):
        # tolerate newly created object types (e.g., EggCracked after break)
        def _find_object_by_type(obj_type):
            base = obj_type.replace("*", "")
            for obj in self.controller.last_event.metadata['objects']:
                if obj['objectType'] == obj_type or obj['objectType'].startswith(base):
                    return obj
            return None

        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            if itemtype not in self.objecttype2object:
                obj = _find_object_by_type(itemtype)
                if obj:
                    self.objecttype2object[itemtype] = [obj]
                else:
                    return None, self.get_legal_navigations(), self.get_legal_interactions()
            item = self.objecttype2object[itemtype][0]

        if item is None:
            item = _find_object_by_type(itemtype)
            if item:
                self.objecttype2object[itemtype] = [item]
            else:
                return None, self.get_legal_navigations(), self.get_legal_interactions()
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["pick_up"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "pick_up",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def put_in(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["put_in"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "put_in",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def toggle(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        if item["isToggled"]==True:
            image_fp, legal_navigations, legal_interactions = None, None, None
            self.action.action_mapping["toggle_off"](self.controller, item['objectId'])
            if itemtype not in self.target_item_type2obj_id:
                self.objecttype2object[itemtype][0]["isToggled"] = False
            image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "toggle",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
            legal_navigations = self.get_legal_navigations()
            legal_interactions = self.get_legal_interactions()
            return image_fp, legal_navigations, legal_interactions
        else:
            image_fp, legal_navigations, legal_interactions = None, None, None
            self.action.action_mapping["toggle_on"](self.controller, item['objectId'])
            if itemtype not in self.target_item_type2obj_id:
                self.objecttype2object[itemtype][0]["isToggled"] = True
            image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "toggle",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
            legal_navigations = self.get_legal_navigations()
            legal_interactions = self.get_legal_interactions()
            return image_fp, legal_navigations, legal_interactions

    def open(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["open"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "open",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions
    
    def close(self, itemtype):
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]
        
        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["close"](self.controller, item['objectId'])
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "close",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions


    def break_obj(self, itemtype):
        """
        Break a target object (AI2-THOR BreakObject). Use when object supports breakable state.
        """
        def _find_object_by_type(obj_type):
            base = obj_type.replace("*", "")
            for obj in self.controller.last_event.metadata['objects']:
                if obj['objectType'] == obj_type or obj['objectType'].startswith(base):
                    return obj
            return None

        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            if itemtype not in self.objecttype2object:
                obj = _find_object_by_type(itemtype)
                if obj:
                    self.objecttype2object[itemtype] = [obj]
                else:
                    return None, self.get_legal_navigations(), self.get_legal_interactions()
            item = self.objecttype2object[itemtype][0]

        if item is None:
            return None, self.get_legal_navigations(), self.get_legal_interactions()

        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["break_"](self.controller, item['objectId'])
        # after break, refresh EggCracked cache if present
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectType'] == "EggCracked":
                self.objecttype2object.setdefault("EggCracked", []).insert(0, obj)
                break
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "break",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions


    def slice_obj(self, itemtype):
        """
        slice a target object (AI2-THOR SliceObject). Use when object supports sliceable state.
        """
        if itemtype in self.target_item_type2obj_id:
            obj_id = self.target_item_type2obj_id[itemtype][0]
            item = self.eventobject.get_object_by_id(self.controller.last_event, obj_id)
        else:
            item = self.objecttype2object[itemtype][0]

        image_fp, legal_navigations, legal_interactions = None, None, None
        self.action.action_mapping["slice_"](self.controller, item['objectId'])
        for obj in self.controller.last_event.metadata['objects']:
            if obj['objectType'].endswith("Sliced"):
                self.objecttype2object.setdefault(obj['objectType'], []).insert(0, obj)
        image_fp = self.save_frame({"step_count": str(self.step_count),
                                    "action": "slice",
                                    "item": item["objectType"]},
                                    prefix_save_path=self.result_dir)
        legal_navigations = self.get_legal_navigations()
        legal_interactions = self.get_legal_interactions()
        return image_fp, legal_navigations, legal_interactions

    def get_all_item_image(self):
        res = []
        for item in tqdm(self.eventobject.get_objects(self.controller.last_event)[0]):
            # print(item["name"],self.eventobject.get_item_surface_area(item['name']))
            if item["name"] == "DiningTable_806ce8fd":#Book_e173324d Box_8e5b2c6b CellPhone_b8be2958
            # # print(item["name"],":",round(item["rotation"]['y']))
                succeess, _ ,_ = self.navigate(item)
                self.save_frame({"item": item["name"]})
                dic = {
                    "scene": self.scene,
                    "item": item["name"],
                    "agent":{
                        "agentMode": "arm",
                        "position": self.get_agent_position(),
                        "rotation": self.get_agent_rotation(),
                    },
                    "camera":{
                        "position": self.get_camera_position(),
                        "rotation": self.get_camera_rotation(),
                    },
                    "fieldOfView":90,
                    "gridSize":0.1,
                    "visibilityDistance": 10,
                    "image_path": f"./data/item_image/{self.scene}_{item['name']}.png"
                }
                res.append(dic)
                self.controller.reset(self.scene)
        with open(f"./data/{self.scene}_objects.jsonl", "w") as f:
            import json
            for item in res:
                f.write(json.dumps(item, ensure_ascii=False)+"\n")

    def get_navigate_path(self):
        res = []
        
        for item in tqdm(self.eventobject.get_objects(self.controller.last_event)[0]):
            # print(item["name"],self.eventobject.get_item_surface_area(item['name']))
            # if item["name"] == "DiningTable_806ce8fd":#Book_e173324d Box_8e5b2c6b CellPhone_b8be2958
            # # print(item["name"],":",round(item["rotation"]['y']))
            # if item["name"] in self.legal_location:
                import copy
                legal_location = copy.deepcopy(self.legal_location)
                succeess, _ ,_ = self.navigate(item)
                visible_objects = []
                obj_names, objs = self.eventobject.get_visible_objects(self.controller.last_event)
                for obj_name, obj in zip(obj_names, objs):
                    volume = self.eventobject.get_item_volume(self.controller.last_event, obj_name)
                    if volume <= 0.1:
                        if obj["distance"] <= 1.5:
                            visible_objects.append(obj["name"])
                    elif volume <= 0.5:
                        if obj["distance"] <= 2.5:
                            visible_objects.append(obj["name"])
                    elif volume <= 1:
                        if obj["distance"] <= 5.0:
                            visible_objects.append(obj["name"])
                    else:
                        if obj["distance"] <= 10.0:
                            visible_objects.append(obj["name"])
                for obj_name in visible_objects:
                    if obj_name not in legal_location.keys():
                        legal_location[obj_name] = 1
                    else:
                        legal_location[obj_name] += 1

                self.save_frame({"item": item["name"]})
                dic = {
                    "scene": self.scene,
                    "init_legal_location": legal_location,
                    "object": {
                        "name":item["name"],
                        "position": item["position"],
                        "rotation": item["rotation"],
                    },
                    "agent":{
                        "agentMode": "default",
                        "position": self.get_agent_position(),
                        "rotation": self.get_agent_rotation(),
                    },
                    "camera":{
                        "position": self.get_camera_position(),
                        "rotation": self.get_camera_rotation(),
                    },
                    "fieldOfView":90,
                    "gridSize":0.1,
                    "visibilityDistance": 10,
                    "image_path": f"./data/item_image/{self.scene}_{item['name']}.png"
                }
                res.append(dic)
                self.controller.reset(self.scene)

        with open(f"./data/{self.scene}_objects.jsonl", "w") as f:
            import json
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False)+"\n")

    def example(self):
        for item in tqdm(self.eventobject.get_objects(self.controller.last_event)[0]):
            if item["name"] == "DiningTable_0beb798c": # Book_e173324d Box_8e5b2c6b CellPhone_b8be2958
                self.navigate(item)
                self.move_observation(item)
                self.adjust_agent_fieldOfView(150)
                self.save_frame({"item": item["name"], "action": "pick_up"})
        
        pass
    
    def test_visibility(self):
        self.init()
        volumes = {
            "0-0.01": [],
            "0.01-0.05": [],
            "0.05-0.1": [],
            "0.1-0.5": [],
            "0.5-1.0": [],
            "1.0+": []
        }
        import json
        with open("./data/visible_objects_f.jsonl") as f:
            data = [json.loads(line) for line in f.readlines()]
        visible_objects = []
        for d in data:
            visible_objects.extend(d["visible_objects"])
            for obj in d["objects"]:
                if not obj["name"].startswith("Floor_"):
                    volume = obj.get("volum")
                    if volume is not None:
                        if 0 <= volume < 0.01:
                            volumes["0-0.01"].append(obj["name"])   
                        elif 0.01 <= volume < 0.05: 
                            volumes["0.01-0.05"].append(obj["name"]) 
                        elif 0.05 <= volume < 0.1:
                            volumes["0.05-0.1"].append(obj["name"])
                        elif 0.1 <= volume < 0.5:
                            volumes["0.1-0.5"].append(obj["name"])
                        elif 0.5 <= volume < 1.0:
                            volumes["0.5-1.0"].append(obj["name"])
                        else:
                            volumes["1.0+"].append(obj["name"])
        
        with open("./data/visible_objects.jsonl", "a") as f:
            import json
            visible_objects = []
            item_names, items = self.eventobject.get_visible_objects(self.controller.last_event)
            for item_name, item in zip(item_names, items):
                volume = self.eventobject.get_item_volume(self.controller.last_event, item_name)
                surface_area = self.eventobject.get_item_surface_area(self.controller.last_event, item_name)
                if volume <= 0.1:
                    if item["distance"] <= 1.5:
                        visible_objects.append(item["name"])
                    elif surface_area > 1:
                        visible_objects.append(item["name"])
                elif volume <= 0.5:
                    if item["distance"] <= 2.5:
                        visible_objects.append(item["name"])
                    elif surface_area > 1:
                        visible_objects.append(item["name"])
                elif volume <= 1:
                    if item["distance"] <= 5.0:
                        visible_objects.append(item["name"])
                    elif surface_area > 1:
                        visible_objects.append(item["name"])
                else:
                    if item["distance"] <= 10.0:
                        visible_objects.append(item["name"])
                        
            dic = {
                "scene": self.scene,
                "objects":[
                    {"name": item["name"], 
                     "visible": item["visible"],
                     "volum": self.eventobject.get_item_volume(self.controller.last_event, item['name']),
                     "surface_area": self.eventobject.get_item_surface_area(self.controller.last_event, item['name']),
                     "distance": item["distance"],
                    }
                for item in self.eventobject.get_objects(self.controller.last_event)[0]
                ],
                "visible_objects": visible_objects,
            }
            f.write(json.dumps(dic, ensure_ascii=False)+"\n")
    
    def get_navigate_location(self):
        metadata = self.controller.last_event.metadata
        volumes = []
        objectid2object={}
        for obj in metadata["objects"]:
            objectid2object[obj["objectId"]]=obj
            if obj["objectType"]!="Floor":
                size=obj["axisAlignedBoundingBox"]["size"]
                # print(size)
                v=size["x"]*size["y"]*size["z"]
                dx=obj["axisAlignedBoundingBox"]["center"]["x"]
                dz=obj["axisAlignedBoundingBox"]["center"]["z"]
                agentx=metadata["agent"]["position"]["x"]
                agentz=metadata["agent"]["position"]["z"]
                d=math.sqrt((dx-agentx)**2+(dz-agentz)**2)
                
                sxz = size["x"] * size["z"]

                sxy = size["x"] * size["y"]

                szy = size["y"] * size["z"]

                s = max(sxz, sxy, szy)
                
                if d != 0:
                    rate = v / d
                else:
                    rate = 0
                    print(obj["objectId"],"d=0")  
                rate=v/d
                isnavigable=False
                if obj["visible"]==True:
                    if v<0.01:
                        isnavigable=False
                        if s>0.5 and d<10:
                            isnavigable=True
                        elif s>0.15 and d<4:
                            isnavigable=True
                        elif s>0.08 and d<2.5:
                            isnavigable=True 
                        elif v>0.005 and d<2:
                            isnavigable=True 
                        elif v>0.001 and d<1.5:
                            isnavigable=True
                        elif d<1:
                            isnavigable=True
                    else:
                        isnavigable=True
                        if rate<=0.02:
                            isnavigable=False
                            if s>0.5 and d<10:
                                isnavigable=True
                            elif s>0.15 and d<4:
                                isnavigable=True
                            elif s>0.08 and d<2.5:
                                isnavigable=True 
                            elif v>0.005 and d<2:
                                isnavigable=True 
                            elif v>0.001 and d<1.5:
                                isnavigable=True
                            elif d<1:
                                isnavigable=True
                                
                volumes.append({
                    "objectId":obj["objectId"],
                    "objectType":obj["objectType"],
                    "visible":obj["visible"],
                    "volume":v,
                    "s":s,
                    "distance":d,
                    "rate":rate,
                    "isnavigable":isnavigable
                })
                sorted_volumes = sorted(volumes, key=lambda v: v["rate"])

        res = {}
        for item in sorted_volumes:
            res[item["objectId"]] = item
        #     if item['isnavigable']:
        #         if item["objectType"] not in self.objecttype2object:
        #             self.objecttype2object[item["objectType"]] = [objectid2object[item["objectId"]]]
        #         else:
        #             itemname = objectid2object[item["objectId"]]['name']
        #             if itemname not in [obj['name'] for obj in self.objecttype2object[item["objectType"]]]:
        #                 self.objecttype2object[item["objectType"]].append(objectid2object[item["objectId"]])
        # print("get_navigate_location:",res)
        return res
    
    def get_legal_navigations(self):
        objects = self.get_navigate_location()
        for objectId, obj in objects.items():
            if obj["isnavigable"]:
                if obj["objectType"] not in self.navigable_objects:
                    self.navigable_objects[obj["objectType"]] = 0
                self.navigable_objects[obj["objectType"]] += 1
        
        return list(self.navigable_objects.keys())

    def get_current_container_obj(self):
        if self.current_container is not None:
            objects = [obj.split("|")[0] for obj in self.current_container["receptacleObjectIds"]]
            # print(objects)
            return objects
        else:
            return []

    def get_legal_interactions(self):
        legal_interactions = {}
        objects = self.get_navigate_location()
        for objectId, obj in objects.items():
            if (obj["visible"] and obj["objectType"] in self.get_current_container_obj()) or obj["isnavigable"]:
                if obj["objectType"] not in legal_interactions:
                    legal_interactions[obj["objectType"]] = 0
                legal_interactions[obj["objectType"]] += 1
        
        self.legal_interactions = legal_interactions
        return list(self.legal_interactions.keys())

    def action_meta(self, navigate_locations, item, action="obervation"):
        if action =="init":
            self.init_agent_corner()
            navigate_location = self.get_navigate_location()
            for k, item in navigate_location.items():
                if item["objectId"] not in navigate_locations:
                    navigate_locations[item["objectId"]] = item
        
        elif action == "obervation":
            for i in range(3):
                self.action.action_mapping["rotate_left"](self.controller, 90)
                navigate_location = self.get_navigate_location()
                for k, item in navigate_location.items():
                    if item["objectId"] not in navigate_locations:
                        navigate_locations[item["objectId"]] = item
        
        elif action == "navigate":
            self.navigate(item)
            navigate_location = self.get_navigate_location()
            for k, item in navigate_location.items():
                if item["objectId"] not in navigate_locations:
                    navigate_locations[item["objectId"]] = item
        
        elif action == "move":
            self.move_forward(0.5)
            navigate_location = self.get_navigate_location()
            for k, item in navigate_location.items():
                if item["objectId"] not in navigate_locations:
                    navigate_locations[item["objectId"]] = item
        
        
        return navigate_locations, navigate_location
    
    def exec(self, action, item=None):
        # for itemtype in self.eventobject.get_objects_type(self.controller.last_event):
        #     if itemtype not in self.navigable_objects:
        #         self.navigable_objects[itemtype] = 0
        #     self.navigable_objects[itemtype] += 1
        #     if itemtype not in self.legal_interactions:
        #         self.legal_interactions[itemtype] = 0
        #     self.legal_interactions[itemtype] += 1
        # print("self.navigable_objects:",self.navigable_objects)
        # print("self.legal_interactions:",self.legal_interactions)
        image_fp, legal_locations, legal_objects = None, list(self.navigable_objects.keys()), list(self.legal_interactions.keys())
        # image_fp, legal_locations, legal_objects = None, self.eventobject.get_objects_type(self.controller.last_event), list(self.legal_interactions.keys())
        self.step_count += 1
        for action_name in self.action_space:
            if action_name in action:
                if action_name == "observe" or action_name == "init":
                    image_fp, legal_locations, legal_objects = self.action_space[action_name]()
                    if self.controller.last_event.metadata["errorMessage"]!="":
                        success=False
                    else:
                        success=True
                    return success, image_fp, legal_locations, legal_objects
                elif action_name == "move forward":
                    image_fp, legal_locations, legal_objects = self.action_space[action_name](distance=0.5)
                    if self.controller.last_event.metadata["errorMessage"]!="":
                        success=False
                    else:
                        success=True
                    return success, image_fp, legal_locations, legal_objects
                else:
                    if item is None:
                        return False, None, list(self.navigable_objects.keys()), list(self.legal_interactions.keys())
                    else:
                        if action_name == "navigate to":
                            # allow navigating to dynamically created objects (e.g., *Sliced / EggCracked)
                            if (
                                item in self.navigable_objects
                                or item in self.target_item_type2obj_id
                                or item in self.objecttype2object
                                or (isinstance(item, str) and item.endswith("Sliced"))
                                or item == "EggCracked"
                            ):
                                image_fp, legal_locations, legal_objects = self.action_space[action_name](item)
                                return True, image_fp, legal_locations, legal_objects
                        if action_name in ["break", "pickup", "slice"]:
                            image_fp, legal_locations, legal_objects = self.action_space[action_name](item)
                            if self.controller.last_event.metadata["errorMessage"]!="":
                                print(self.controller.last_event.metadata["errorMessage"])
                                success=False
                            else:
                                success=True
                            return success, image_fp, legal_locations, legal_objects
                        if action_name in ["put", "put in","toggle", "open", "close"] and item in self.legal_interactions:
                            image_fp, legal_locations, legal_objects = self.action_space[action_name](item)
                            if self.controller.last_event.metadata["errorMessage"]!="":
                                print(self.controller.last_event.metadata["errorMessage"])
                                success=False
                            else:
                                success=True
                            return success, image_fp, legal_locations, legal_objects
                
        
        return False, image_fp, legal_locations, legal_objects


if __name__ == "__main__":
    autogn = RocAgent("", visibilityDistance=10, fieldOfView=90)
    autogn.get_all_item_image()
    autogn.example()
    autogn.init_agent_corner()
    autogn.test_visibility()
    autogn.get_navigate_path()
    autogn.controller.stop()
    
