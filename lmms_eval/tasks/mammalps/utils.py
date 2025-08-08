import json
import re
import os
from typing import Dict, List, Optional

from loguru import logger as eval_logger


def mammalps_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Extract video path from document and download from HuggingFace if needed"""
    clip_path = doc["clip"]
    
    # If it's already an absolute path and exists, use it
    if os.path.isabs(clip_path) and os.path.exists(clip_path):
        return [clip_path]
    
    # Download from HuggingFace dataset repository
    try:
        from huggingface_hub import hf_hub_download
        
        # Download the video file from the HuggingFace dataset repository
        local_path = hf_hub_download(
            repo_id="luciehmct/mammalps-test-recognition",
            filename=clip_path,
            repo_type="dataset"
        )
        
        if os.path.exists(local_path):
            return [local_path]
        else:
            eval_logger.error(f"Downloaded file does not exist: {local_path}")
            return [clip_path]
            
    except Exception as e:
        eval_logger.error(f"Failed to download video {clip_path}: {str(e)}")
        return [clip_path]


def mammalps_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract the human prompt from conversations"""
    if "conversations" in doc and len(doc["conversations"]) > 0:
        human_message = doc["conversations"][0]
        if human_message.get("from") == "human":
            return human_message["value"]
    
    # Fallback to generic prompt if conversations not found
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    subtask = lmms_eval_specific_kwargs.get("subtask", "animal")
    
    
    # DEBUG: Log what subtask we're generating prompt for with clip and question info
    # eval_logger.info(f"DEBUG: doc_to_text called with subtask: {subtask}")
    # eval_logger.info(f"DEBUG: doc_to_text kwargs: {lmms_eval_specific_kwargs}")
    # eval_logger.info(f"DEBUG: doc_to_text - clip_path: {doc.get('clip', 'Unknown')}")
    # eval_logger.info(f"DEBUG: doc_to_text - question_id: {doc.get('id', 'Unknown')}")

    ## PROBABILY NOT NEEDED
    # import inspect
    # detected_subtask = subtask
    
    # try:
    #     frame = inspect.currentframe()
    #     while frame:
    #         frame_locals = frame.f_locals
            
    #         # Look for task-related variables in the call stack
    #         if 'task' in frame_locals:
    #             task_obj = frame_locals['task']
    #             if hasattr(task_obj, '_config') and hasattr(task_obj._config, 'task'):
    #                 task_name = task_obj._config.task
    #                 eval_logger.info(f"DEBUG: Found task name in doc_to_text: {task_name}")
    #                 if 'action' in task_name:
    #                     detected_subtask = 'action'
    #                 elif 'activity' in task_name:
    #                     detected_subtask = 'activity'
    #                 elif 'animal' in task_name:
    #                     detected_subtask = 'animal'
    #                 break
            
    #         # Additional check for task name in frame
    #         if 'self' in frame_locals and hasattr(frame_locals['self'], '_config'):
    #             config = frame_locals['self']._config
    #             if hasattr(config, 'task'):
    #                 task_name = config.task
    #                 eval_logger.info(f"DEBUG: Found task name string in doc_to_text: {task_name}")
    #                 if 'action' in task_name:
    #                     detected_subtask = 'action'
    #                 elif 'activity' in task_name:
    #                     detected_subtask = 'activity'
    #                 elif 'animal' in task_name:
    #                     detected_subtask = 'animal'
            
    #         frame = frame.f_back
    # except Exception as e:
    #     eval_logger.warning(f"DEBUG: Could not extract task from stack in doc_to_text: {e}")
    
    # Use the detected subtask
    # subtask = detected_subtask
    eval_logger.info(f"DEBUG: Final determined subtask for prompt: {subtask}")
    
    # Common prompt template
    def _generate_mammalps_prompt(entity_type, entity_plural, definition, label_space, question):
        """Generate a consistent prompt for mammalps subtasks"""
        return f"""You are an assistant specialized in analyzing animal videos. Your task is to answer questions about the animals and their behaviors in a given video.

**Instruction:**
You are provided with the following base function, which you can use to decompose the main question into subtasks and solve them step by step:
def recognize(entity_type: str, condition: Optional[str]) -> List[str]:
    \"\"\"Returns all unique entities of the specified type detected in the video (e.g., 'animal', 'action', 'activity').
    If condition is provided, returns all entities of the specified type that appear when the given condition is true.

    Example:
        >>> recognize(entity_type='animal')
        ['dog', 'cat']
        >>> recognize(entity_type='action')
        ['bark', 'run']
        >>> recognize(entity_type='action', condition='animal == dog')
        ['bark', 'run']
    \"\"\"
In addition to these base function, you may use standard Python functions such as average, max, min, sum, len, sorted, etc., as needed to help you answer the questions.

**Output format:**
Your final output should be 'Final answer:' followed by the list of {entity_plural} recognized in the video, formatted as List[str].

{definition}
You should use the following label space to identify {entity_plural}:
{entity_type} label space: {label_space}

<video>
{question}"""
    
    if subtask == "animal":
        return _generate_mammalps_prompt(
            entity_type="animals",
            entity_plural="animals", 
            definition="",
            label_space="roe_deer, fox, red_deer, wolf, hare",
            question="Find all animals that are present in the video footage.\n\nYour answer should follow the example below:\nstep 1\nanimals = recognize(entity_type='animal')\noutput:List[str]: ['red_deer']\n\nstep 2\nreturn animals\noutput:Final answer: ['red_deer']"
        )
    
    elif subtask == "action":
        return _generate_mammalps_prompt(
            entity_type="actions",
            entity_plural="actions",
            definition="An action is a discrete, often well-defined motor event or behavior performed by an animal, typically characterized by a specific goal or function.",
            label_space="sniffing, looking_at_camera, scratching_hoof, grazing, running, drinking, shaking_fur, jumping, unknown, bathing, urinating, scratching_body, standing_head_up, scratching_antlers, vocalizing, laying, standing_head_down, defecating, walking",
            question="What actions do the animals perform during the video?\n\nYour answer should follow the example below:\nstep 1\nactions = recognize(entity_type='action')\noutput:List[str]: ['eating', 'attending']\n\nstep 2\nreturn actions\noutput:Final answer: ['eating', 'attending']"
        )
    
    elif subtask == "activity":
        return _generate_mammalps_prompt(
            entity_type="activities",
            entity_plural="activities",
            definition="An activity is a broader or longer-lasting pattern of behavior, often encompassing multiple actions that together form a functional behavioral state or mode.",
            label_space="marking, unknown, camera_reaction, grooming, foraging, chasing, playing, escaping, vigilance, resting, courtship",
            question="Detect all animal activities occurring in the video.\n\nYour answer should follow the example below:\nstep 1\nactivities = recognize(entity_type='activity')\noutput:List[str]: ['foraging']\n\nstep 2\nreturn activities\noutput:Final answer: ['foraging']"
        )
    
    return ""


def mammalps_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    """Extract target answer based on subtask"""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    subtask = lmms_eval_specific_kwargs.get("subtask", "animal")
    
    # DEBUG: Log what subtask we're extracting for with clip and question info
    # eval_logger.info(f"DEBUG: doc_to_target called with subtask: {subtask}")
    # eval_logger.info(f"DEBUG: doc_to_target kwargs: {lmms_eval_specific_kwargs}")
    # eval_logger.info(f"DEBUG: doc has keys: {list(doc.keys())}")
    # eval_logger.info(f"DEBUG: clip_path: {doc.get('clip', 'Unknown')}")
    # eval_logger.info(f"DEBUG: question_id: {doc.get('id', 'Unknown')}")
    
    # Try to detect the subtask from the call stack
    # since lmms_eval_specific_kwargs may not contain the subtask
    import inspect
    detected_subtask = subtask
    
    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals
            
            # Look for task-related variables in the call stack
            if 'task' in frame_locals:
                task_obj = frame_locals['task']
                if hasattr(task_obj, '_config') and hasattr(task_obj._config, 'task'):
                    task_name = task_obj._config.task
                    eval_logger.info(f"DEBUG: Found task name: {task_name}")
                    if 'action' in task_name:
                        detected_subtask = 'action'
                    elif 'activity' in task_name:
                        detected_subtask = 'activity'
                    elif 'animal' in task_name:
                        detected_subtask = 'animal'
                    break
            
            # Additional check for task name in frame
            if 'self' in frame_locals and hasattr(frame_locals['self'], '_config'):
                config = frame_locals['self']._config
                if hasattr(config, 'task'):
                    task_name = config.task
                    eval_logger.info(f"DEBUG: Found task name string: {task_name}")
                    if 'action' in task_name:
                        detected_subtask = 'action'
                    elif 'activity' in task_name:
                        detected_subtask = 'activity'
                    elif 'animal' in task_name:
                        detected_subtask = 'animal'
            
            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"DEBUG: Could not extract task from stack in doc_to_target: {e}")
    
    eval_logger.info(f"DEBUG: Final determined subtask: {detected_subtask}")
    
    if detected_subtask == "animal":
        target = doc.get("animal", [])
        eval_logger.info(f"DEBUG: Returning animal target: {target}")
        return target
    elif detected_subtask == "action":
        target = doc.get("action", [])
        eval_logger.info(f"DEBUG: Returning action target: {target}")
        return target
    elif detected_subtask == "activity":
        target = doc.get("activity", [])
        eval_logger.info(f"DEBUG: Returning activity target: {target}")
        return target
    
    eval_logger.warning(f"DEBUG: Unknown subtask '{detected_subtask}', defaulting to animal")
    return doc.get("animal", [])


def mammalps_jaccard_metric(predictions, references):
    """Custom jaccard metric that parses mammalps format responses"""
    from lmms_eval.tasks.megabench.metrics.scoring.common.metrics import jaccard_index
    
    # Parse the prediction if it's a string with "Final answer:" format
    if isinstance(predictions, list) and len(predictions) > 0:
        response = predictions[0]
    else:
        response = predictions
    
    if isinstance(response, str):
        # Extract list from "Final answer: [...]" format
        predicted_labels = []
        
        # Try to extract from "Final answer: ['item1', 'item2']" format
        final_answer_match = re.search(r"Final answer:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            try:
                predicted_labels = eval(final_answer_match.group(1))
                if isinstance(predicted_labels, list):
                    predicted_labels = [str(label).strip() for label in predicted_labels]
                    eval_logger.info(f"DEBUG: Extracted via final_answer format: {predicted_labels}")
            except Exception as e:
                eval_logger.warning(f"DEBUG: Failed to eval final_answer match: {e}")
        
        # Fallback: try to extract any list-like structure
        if not predicted_labels:
            list_match = re.search(r"\[([^\]]+)\]", response)
            if list_match:
                content = list_match.group(1)
                # Split by comma and clean up
                predicted_labels = [item.strip().strip("'\"") for item in content.split(",")]
                eval_logger.info(f"DEBUG: Fallback - Extracted via list format: {predicted_labels}")
        
        # Final fallback: try to extract individual quoted items
        if not predicted_labels:
            quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", response)
            if quoted_items:
                predicted_labels = quoted_items
                eval_logger.info(f"DEBUG: Fallback - Extracted via quoted format: {predicted_labels}")
        
        if not predicted_labels:
            eval_logger.warning(f"DEBUG: No labels extracted from response! Using empty list.")
            predicted_labels = []
        
        parsed_prediction = predicted_labels
    else:
        # If predictions is already a list, use it directly
        parsed_prediction = predictions if isinstance(predictions, list) else [predictions]
    
    # Calculate jaccard index
    jaccard_score = jaccard_index(parsed_prediction, references)
    eval_logger.info(f"DEBUG: Jaccard calculation: pred={parsed_prediction}, ref={references}, score={jaccard_score}")
    
    return {"mammalps_jaccard": jaccard_score}


def mammalps_process_results(doc, results, lmms_eval_specific_kwargs=None):
    """Process model results to extract lists from 'Final answer:' format"""
    
    if not results or len(results) == 0:
        eval_logger.warning("No results provided to process_results")
        return []
    
    response = results[0].strip()
    subtask = lmms_eval_specific_kwargs.get("subtask", "animal") if lmms_eval_specific_kwargs else "animal"
    
    # DEBUG: Log clip_path and question_id for extraction from logs
    eval_logger.info(f"DEBUG: process_results - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.info(f"DEBUG: process_results - full_model_response: {repr(response)}")
    
    # Try to detect the subtask from the call stack
    import inspect
    detected_subtask = subtask
    
    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals
            
            # Look for task-related variables in the call stack
            if 'task' in frame_locals:
                task_obj = frame_locals['task']
                if hasattr(task_obj, '_config') and hasattr(task_obj._config, 'task'):
                    task_name = task_obj._config.task
                    eval_logger.info(f"DEBUG: Found task name in process_results: {task_name}")
                    if 'action' in task_name:
                        detected_subtask = 'action'
                    elif 'activity' in task_name:
                        detected_subtask = 'activity'
                    elif 'animal' in task_name:
                        detected_subtask = 'animal'
                    break
            
            # Additional check for task name in frame
            if 'self' in frame_locals and hasattr(frame_locals['self'], '_config'):
                config = frame_locals['self']._config
                if hasattr(config, 'task'):
                    task_name = config.task
                    eval_logger.info(f"DEBUG: Found task name string in process_results: {task_name}")
                    if 'action' in task_name:
                        detected_subtask = 'action'
                    elif 'activity' in task_name:
                        detected_subtask = 'activity'
                    elif 'animal' in task_name:
                        detected_subtask = 'animal'
            
            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"DEBUG: Could not extract task from stack in process_results: {e}")
    
    # Use the detected subtask
    subtask = detected_subtask
    
    # DEBUG: Log the actual model response with clip and question info
    eval_logger.info(f"DEBUG: Processing model response for subtask: {subtask}")
    eval_logger.info(f"DEBUG: clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.info(f"DEBUG: question_id: {doc.get('id', 'Unknown')}")
    eval_logger.info(f"DEBUG: Full response: {repr(response)}")
    
    # Extract list from response - look for "Final answer:" pattern first
    predicted_labels = []
    
    # Try to extract from "Final answer: ['item1', 'item2']" format
    final_answer_match = re.search(r"Final answer:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        try:
            predicted_labels = eval(final_answer_match.group(1))
            if isinstance(predicted_labels, list):
                predicted_labels = [str(label).strip() for label in predicted_labels]
                eval_logger.info(f"DEBUG: Extracted via final_answer format: {predicted_labels}")
        except Exception as e:
            eval_logger.warning(f"DEBUG: Failed to eval final_answer match: {e}")
    
    # Fallback: try to extract any list-like structure
    if not predicted_labels:
        list_match = re.search(r"\[([^\]]+)\]", response)
        if list_match:
            content = list_match.group(1)
            # Split by comma and clean up
            predicted_labels = [item.strip().strip("'\"") for item in content.split(",")]
            eval_logger.info(f"DEBUG: Fallback - Extracted via list format: {predicted_labels}")
    
    # Final fallback: try to extract individual quoted items
    if not predicted_labels:
        quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", response)
        if quoted_items:
            predicted_labels = quoted_items
            eval_logger.info(f"DEBUG: Fallback - Extracted via quoted format: {predicted_labels}")
    
    if not predicted_labels:
        eval_logger.warning(f"DEBUG: No labels extracted from response! Returning empty list.")
        predicted_labels = []
    
    eval_logger.info(f"DEBUG: Final extracted labels: {predicted_labels}")
    
    # Compute jaccard score directly since we have both prediction and target
    def jaccard_index(pred_set, target_set):
        """Calculate Jaccard index between two sets"""
        pred_set = set(pred_set) if pred_set else set()
        target_set = set(target_set) if target_set else set()
        
        if len(pred_set) == 0 and len(target_set) == 0:
            return 1.0
        
        intersection = len(pred_set.intersection(target_set))
        union = len(pred_set.union(target_set))
        
        return intersection / union if union > 0 else 0.0
    
    # Get the target for comparison
    target_key = "animal"  # Default to animal
    if subtask:
        target_key = subtask
    
    target_labels = doc.get(target_key, [])
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    
    # Calculate jaccard score
    jaccard_score = jaccard_index(predicted_labels, target_labels)
    eval_logger.info(f"DEBUG: Jaccard score: pred={predicted_labels}, target={target_labels}, score={jaccard_score}")
    eval_logger.info(f"DEBUG: Final result for clip_path={doc.get('clip', 'Unknown')}, question_id={doc.get('id', 'Unknown')}, subtask={subtask}")
    
    # Return the computed jaccard score - the aggregation function will average these
    return {
        "jaccard": jaccard_score
    }

# Aggregation function for MammalPS jaccard results
def mammalps_jaccard_aggregation(items): 
    """Aggregate jaccard results - items contains the values from process_results"""
    if not items:
        return 0.0
    
    def jaccard_index(pred_set, target_set):
        """Calculate Jaccard index between two sets"""
        pred_set = set(pred_set) if pred_set else set()
        target_set = set(target_set) if target_set else set()
        
        if len(pred_set) == 0 and len(target_set) == 0:
            return 1.0
        
        intersection = len(pred_set.intersection(target_set))
        union = len(pred_set.union(target_set))
        
        return intersection / union if union > 0 else 0.0
    
    # The aggregation function receives the results from the metric computation
    # Each item should be the jaccard score computed by the framework
    total_jaccard = 0.0
    count = 0
    
    for item in items:
        if isinstance(item, (int, float)):
            # This is a computed jaccard score
            total_jaccard += item
            count += 1
        elif isinstance(item, list):
            # This should not happen with the proper setup, but handle it anyway
            eval_logger.warning(f"DEBUG: Aggregation received raw list: {item}")
        else:
            eval_logger.warning(f"DEBUG: Unexpected item type in aggregation: {type(item)}, value: {item}")
    
    return total_jaccard / count if count > 0 else 0.0
    
    for result in results:
        predicted = set(result.get("predicted", []))
        target = set(result.get("target", []))
        
        # Calculate Jaccard index
        intersection = len(predicted.intersection(target))
        union = len(predicted.union(target))
        
        if union == 0:
            jaccard = 1.0 if len(predicted) == 0 and len(target) == 0 else 0.0
        else:
            jaccard = intersection / union
        
        total_jaccard += jaccard
        count += 1
    
    return total_jaccard / count if count > 0 else 0.0