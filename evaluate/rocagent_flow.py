import os
import sys
import json
import argparse
import time
from typing import List, Dict
from tqdm import tqdm
from ai2thor.controller import Controller
from ai2thor_engine.RocAgent import RocAgent

MODE = "LOCAL"
PLATFORM_TYPE = "GPU"

LOG_FILE_PATH = "./results/execution_errors.json"

def load_data(args):
    file_path = args.input_path
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_and_save_container_info(autogn, task_id, task_name, tasktype, related_objects, save_path):
    first_related_object = related_objects[0]
    print(f" {first_related_object}")
    
    container_name = None
    
    try:
        item = autogn.eventobject.get_object_by_id(autogn.controller.last_event, first_related_object)
        print(f"{item}")
        container_name = item['parentReceptacles'][0].split('|')[0]
        print(container_name)
    except Exception as e:
        print(f" {e}")
    
    return container_name

def record_error(task_id, error_message):
    try:
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
            
        error_log = {
            "identity": task_id,
            "error_message": str(error_message),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logs.append(error_log)
        
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
            
        print(f" {LOG_FILE_PATH}")
    except Exception as e:
        print(f"{e}")

def test(controller, test_data, input_path=None, save_image=True):
    scene = test_data.get('scene', 'FloorPlan1')
    tasktype = test_data.get('tasktype', '')
    instruction_idx = test_data.get('instruction_idx', '')
    
    task_id = str(test_data.get('identity', 'unknown'))
        
    task_name = test_data.get('taskname', '') or test_data.get('name', '')
    if not task_name:
        task_name = f"unknown_task_{task_id}"
        print(f"{task_id}：{task_name}")
    
    save_path = f"./results/test_data_generation_{task_id}_{task_name.replace(' ', '_')}"
    try:
        os.makedirs(save_path, exist_ok=True)
        print(f" {save_path}")
    except Exception as e:
        print(f"{e}")
        fallback_path = f"./results/{task_id}"
        os.makedirs(fallback_path, exist_ok=True)
        save_path = fallback_path
        print(f"{save_path}")
    
    if input_path:
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        filename_without_ext = os.path.splitext(input_filename)[0]
        error_log_filename = f"{filename_without_ext}_errorLog.json"
        error_log_path = os.path.join(input_dir, error_log_filename)
    else:
        error_log_path = "./results/execution_errors.json"
    
    target_objects = test_data.get("target_objects", [])
    related_objects = test_data.get("related_objects", [])
    navigable_objects = test_data.get("navigable_objects", [])
    print(f"target_objects: {target_objects}")
    print(f"related_objects: {related_objects}")
    print(f"navigable_objects: {navigable_objects}")
    
    autogn = RocAgent(controller, save_path, scene, visibilityDistance=20, gridSize=0.1, fieldOfView=90,
                     target_objects=target_objects,
                     related_objects=related_objects,
                     navigable_objects=navigable_objects,
                     taskid=task_id,
                     platform_type=PLATFORM_TYPE)
    
            # NOTE: Any InitialRandomSpawn done before RocAgent init will be reset by controller.reset in BaseAgent.
        # If you need extra objects, spawn them AFTER RocAgent is created so they persist.
    try:
        spawn_event2 = autogn.controller.step(
            action="InitialRandomSpawn",
            randomSeed=0,
            forceVisible=False,
            numPlacementAttempts=10,
            placeStationary=True,
            numDuplicatesOfType=[
                {"objectType": "Apple", "count": 2}
            ],
            excludedReceptacles=["SinkBasin", "Cabinet"],
            excludedObjectIds=[ 

                ]
        )
        # Debug: print count of Book before/after if needed
        print("[Spawn-after-init] lastActionSuccess:", spawn_event2.metadata.get("lastActionSuccess"),
                "error:", spawn_event2.metadata.get("errorMessage", ""))
    except Exception as e:
            print("[Spawn-after-init] exception:", e)
    # if rotate_objects:
    #         rotate_objects_before_evaluation(autogn, task)
    
    try:
        actions = []
        items = []
        
        if 'task_metadata' in test_data:
            scene_metadata = test_data['task_metadata']
            for a in scene_metadata['actions']:
                if a['action'].lower() == 'end':
                    continue
                actions.append(a['action'])
                items.append(a['objectType'])
        else:
            with open(f"data/single_search_task_metadata/{test_data['scene']}.json") as f:
                scene_metadata = json.load(f)[0]
            for a in scene_metadata[id]['actions']:
                if a['action'].lower() == 'end':
                    continue
                actions.append(a['action'])
                items.append(a['objectType'])
        
        print("actions:", actions)
        print("items:", items)
        
        task_has_error = False
        for i in range(len(actions)):
            action = actions[i]
            item = items[i]
            
            print(f"{i+1}/{len(actions)}: {action} {item}")
            
            success, image_fp, legal_locations, legal_objects = autogn.exec(action, item)
            
            if not success:
                print(f"{action} {item} ")
                task_has_error = True
                
                try:
                    error_log_dir = os.path.dirname(error_log_path)
                    if error_log_dir and not os.path.exists(error_log_dir):
                        os.makedirs(error_log_dir, exist_ok=True)
                        
                    if os.path.exists(error_log_path):
                        with open(error_log_path, "r", encoding="utf-8") as f:
                            error_logs = json.load(f)
                    else:
                        error_logs = []
                        
                    error_record = {
                        "identity": task_id,
                        "task_name": task_name,
                        "action_index": i + 1,
                        "total_actions": len(actions),
                        "failed_action": action,
                        "failed_item": item,
                        "error_message": f" {action} {item} ",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    error_logs.append(error_record)
                    
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        json.dump(error_logs, f, ensure_ascii=False, indent=2)
                        
                    print(f" {error_log_path}")
                except Exception as e:
                    print(f"{e}")
                
                record_error(task_id, f"{action} {item} ")
        
        if not task_has_error:
            print(f" {task_id} ")
            
            if input_path:
                input_dir = os.path.dirname(input_path)
                input_filename = os.path.basename(input_path)
                filename_without_ext = os.path.splitext(input_filename)[0]
                success_filename = f"{filename_without_ext}_success.json"
                success_path = os.path.join(input_dir, success_filename)
            else:
                success_path = "./results/execution_success.json"
            
            try:
                success_dir = os.path.dirname(success_path)
                if success_dir and not os.path.exists(success_dir):
                    os.makedirs(success_dir, exist_ok=True)
                    
                if os.path.exists(success_path):
                    with open(success_path, "r", encoding="utf-8") as f:
                        success_tasks = json.load(f)
                else:
                    success_tasks = []
                    
                success_tasks.append(test_data)
                
                with open(success_path, "w", encoding="utf-8") as f:
                    json.dump(success_tasks, f, ensure_ascii=False, indent=2)
                    
                print(f" {success_path}")
            except Exception as e:
                print(f"{e}")
        else:
            print(f" {task_id} ")
            
            if input_path:
                input_dir = os.path.dirname(input_path)
                input_filename = os.path.basename(input_path)
                filename_without_ext = os.path.splitext(input_filename)[0]
                failed_filename = f"{filename_without_ext}_failed.json"
                failed_path = os.path.join(input_dir, failed_filename)
            else:
                failed_path = "./results/execution_failed.json"
            
            try:
                failed_dir = os.path.dirname(failed_path)
                if failed_dir and not os.path.exists(failed_dir):
                    os.makedirs(failed_dir, exist_ok=True)
                    
                if os.path.exists(failed_path):
                    with open(failed_path, "r", encoding="utf-8") as f:
                        failed_tasks = json.load(f)
                else:
                    failed_tasks = []
                    
                failed_tasks.append(test_data)
                
                with open(failed_path, "w", encoding="utf-8") as f:
                    json.dump(failed_tasks, f, ensure_ascii=False, indent=2)
                    
                print(f" {failed_path}")
            except Exception as e:
                print(f" {e}")
    finally:
        print(f"{task_id} ")
        if 'autogn' in locals():
            try:
                time.sleep(0.1)
                print(f"autogn")
                del autogn
                print(f"autogn")
            except Exception as e:
                print(f"{e}")
                record_error(task_id, f"{str(e)}")
        
        if not save_image and os.path.exists(save_path):
            try:
                import shutil
                print(f"{save_path}")
                shutil.rmtree(save_path)
                print(f"{save_path}")
            except Exception as e:
                print(f" {e}")
                record_error(task_id, f"{str(e)}")

def main():
    if MODE == "LOCAL":
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="./data/FloorPlan1.json", help="input file path")
        parser.add_argument("--cur_count", type=int, default=1, help="")
        parser.add_argument("--total_count", type=int, default=4, help="")
        parser.add_argument("--save_image", type=bool, default=True)
        args = parser.parse_args()
        print(args)
        data = load_data(args)
        success_count = 0
        print("********* start evaluate *********")
        controller = Controller(
            snapToGrid=False,
            quality='Medium',
            agentMode="default",
            massThreshold=None,
            scene='FloorPlan1',
            visibilityDistance=20,
            gridSize=0.1,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=800,
            height=450,
            fieldOfView=90,
        )
        
        print("********* controller init success *********")
        
        input_dir = os.path.dirname(args.input_path)
        input_filename = os.path.basename(args.input_path)
        filename_without_ext = os.path.splitext(input_filename)[0]
        error_log_filename = f"{filename_without_ext}_errorLog.json"
        error_log_path = os.path.join(input_dir, error_log_filename)
        
        for test_data in tqdm(data):
            try:
                test(controller, test_data, args.input_path, save_image=args.save_image)
                success_count += 1
            except Exception as e:
                task_id = str(test_data.get('identity', 'unknown'))
                task_name = test_data.get('taskname', '') or test_data.get('name', 'unknown_task')
                print(f"{str(e)}")
                print(f"--task{task_id}failed, End the current evaluation task!!!--")
                
                try:
                    error_log_dir = os.path.dirname(error_log_path)
                    if error_log_dir and not os.path.exists(error_log_dir):
                        os.makedirs(error_log_dir, exist_ok=True)
                        
                    if os.path.exists(error_log_path):
                        with open(error_log_path, "r", encoding="utf-8") as f:
                            error_logs = json.load(f)
                    else:
                        error_logs = []
                        
                    error_record = {
                        "identity": task_id,
                        "task_name": task_name,
                        "error_type": "task_execution_exception",
                        "error_message": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    error_logs.append(error_record)
                    
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        json.dump(error_logs, f, ensure_ascii=False, indent=2)
                        
                    print(f"{error_log_path}")
                except Exception as log_error:
                    print(f"{log_error}")
                
                record_error(task_id, str(e))
                continue
        
        print(f"--The current process evaluation task end--total task count:{len(data)}successed task count:{success_count}")
        
        print("Controller")
        controller.stop()

if __name__ == "__main__":
    main()