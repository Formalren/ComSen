#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    print(f"related_objects: {first_related_object}")
    
    container_name = None
    
    try:
        item = autogn.eventobject.get_object_by_id(autogn.controller.last_event, first_related_object)
        print(f": {item}")
        container_name = item['parentReceptacles'][0].split('|')[0]
        print(container_name)
    except Exception as e:
        print(f": {e}")
    
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
            
        print(f": {LOG_FILE_PATH}")
    except Exception as e:
        print(f": {e}")


def test(controller, test_data, input_path=None, save_image=True):
    scene = test_data.get('scene', 'FloorPlan1')
    tasktype = test_data.get('tasktype', '')
    instruction_idx = test_data.get('instruction_idx', '')
    
    task_id = str(test_data.get('identity', 'unknown'))
        
    task_name = test_data.get('taskname', '') or test_data.get('name', '')
    if not task_name:
        task_name = f"unknown_task_{task_id}"
        print(f": {task_id}，: {task_name}")
    
    save_path = f"./results/test_data_generation_{task_id}_{task_name.replace(' ', '_')}"
    try:
        os.makedirs(save_path, exist_ok=True)
        print(f": {save_path}")
    except Exception as e:
        print(f": {e}")
        fallback_path = f"./results/{task_id}"
        os.makedirs(fallback_path, exist_ok=True)
        save_path = fallback_path
        print(f": {save_path}")
    
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
    
    initial_position = autogn.get_agent_position()
    initial_rotation = autogn.get_agent_rotation()
    initial_horizon = autogn.get_agent_horizon()
    initial_standing = autogn.controller.last_event.metadata['agent']['isStanding']
    
    try:
        has_pre_handle = 'pre_handle' in test_data and test_data['pre_handle'] is not None
        
        if has_pre_handle:
            print("pre_handle，...")
            pre_handle_data = test_data['pre_handle']

            pre_handle_success = execute_actions(autogn, pre_handle_data['task_metadata'], 
                                               task_id, task_name, error_log_path, "pre-task")

            if not pre_handle_success:
                print("，")
                raise Exception(f"Task {task_id} pre-task execution failed")

            print("，...")
            try:
                teleport_result = controller.step(
                    action="Teleport",
                    position=initial_position,
                    rotation=initial_rotation,
                    horizon=initial_horizon,
                    standing=initial_standing
                )
                if teleport_result.metadata['lastActionSuccess']:
                    print("")
                    try:
                        teleport_image_path = autogn.save_frame({
                            "action": "teleport_after_pre_handle",
                            "task_id": task_id,
                            "timestamp": time.strftime("%Y%m%d_%H%M%S")
                        }, prefix_save_path=save_path)
                        print(f": {teleport_image_path}")
                    except Exception as e:
                        print(f": {e}")
                        record_error(task_id, f"Failed to save the post-pre-task teleport image: {str(e)}")
                else:
                    print(f": {teleport_result.metadata['errorMessage']}")
                    record_error(task_id, f"Agent teleport failed: {teleport_result.metadata['errorMessage']}")
            except Exception as e:
                print(f": {e}")
                record_error(task_id, f"Teleport operation failed: {str(e)}")

        print("...")
        main_task_success = execute_actions(autogn, test_data['task_metadata'], 
                                          task_id, task_name, error_log_path, "main task")

        if not main_task_success:
            print("")
            raise Exception(f"Task {task_id} main task execution failed")

        print(f" {task_id} ")

        print("，")
    finally:
        print(f" {task_id} ")
        if 'autogn' in locals():
            try:
                time.sleep(0.1)
                print(f"autogn")
                del autogn
                print(f"autogn")
            except Exception as e:
                print(f": {e}")
                record_error(task_id, f"Resource cleanup failed: {str(e)}")

        if not save_image and os.path.exists(save_path):
            try:
                import shutil
                print(f": {save_path}")
                shutil.rmtree(save_path)
                print(f": {save_path}")
            except Exception as e:
                print(f": {e}")
                record_error(task_id, f"Failed to delete the image directory: {str(e)}")



def main():
    if MODE == "LOCAL":
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_path", type=str, default="", help="input file path")
        parser.add_argument("--cur_count", type=int, default=1, help="")
        parser.add_argument("--total_count", type=int, default=4, help="")
        parser.add_argument("--save_image", type=bool, default=True, help="Save generated images")
        parser.add_argument("--remove_failed", type=bool, default=True, help="Save successful and failed tasks separately")
        args = parser.parse_args()
        print(args)
        data = load_data(args)
        success_count = 0
        failed_identities = []
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
                print(f": {str(e)}")
                print(f"--task{task_id}failed, End the current evaluation task!!!--")
                
                failed_identities.append(task_id)
                
                try:
                    error_log_dir = os.path.dirname(error_log_path)
                    if error_log_dir and not os.path.exists(error_log_dir):
                        os.makedirs(error_log_dir, exist_ok=True)
                        
                    if os.path.exists(error_log_path):
                        with open(error_log_path, "r", encoding="utf-8") as f:
                            error_logs = json.load(f)
                    else:
                        error_logs = []
                        
                    related_objects_entities = []
                    try:
                        related_objects = test_data.get("related_objects", [])
                        if related_objects:
                            for obj_id in related_objects:
                                try:
                                    related_objects_entities.append({
                                        "object_id": obj_id,
                                        "object_type": "unknown"
                                    })
                                except Exception as obj_error:
                                    print(f"related_object {obj_id} : {obj_error}")
                                    related_objects_entities.append({
                                        "object_id": obj_id,
                                        "object_type": "error_fetching"
                                    })
                    except Exception as related_error:
                        print(f"related_objects: {related_error}")
                        related_objects_entities = []
                    
                    error_record = {
                        "identity": task_id,
                        "task_name": task_name,
                        "error_type": "task_execution_exception",
                        "error_message": str(e),
                        "related_objects": related_objects_entities,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    error_logs.append(error_record)
                    
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        json.dump(error_logs, f, ensure_ascii=False, indent=2)
                        
                    print(f": {error_log_path}")
                    print(f"{len(related_objects_entities)} ")
                except Exception as log_error:
                    print(f" {log_error}")
                
                record_error(task_id, str(e))
                continue
        
        print(f"--The current process evaluation task end--total task count:{len(data)}successed task count:{success_count}")
        
        if args.remove_failed:
      
            success_file_path, failure_file_path = save_success_and_failure_tasks(data, failed_identities, args.input_path)
        
        else:
            print("")
        
        controller.stop()

if __name__ == "__main__":
    main()
