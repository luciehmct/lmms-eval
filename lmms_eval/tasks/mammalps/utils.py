import re
import os
import json

from loguru import logger as eval_logger


def mammalps_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Extract visual information from the document, downloading it from HuggingFace if necessary.

    Args:
        doc (dict): A dictionary representing the document, expected to contain a "clip" key with the path or filename of the video file.
        lmms_eval_specific_kwargs (dict, optional): Additional keyword arguments specific to LMMS evaluation. Defaults to None.

    Returns:
        list[str]: A list containing the local file path to the video clip. If the file cannot be found or downloaded, returns the original path.
    """
    clip_path = doc["clip"]

    # If it's already an absolute path and exists, use it
    if os.path.isabs(clip_path) and os.path.exists(clip_path):
        return [clip_path]

    # Download from HuggingFace dataset repository
    try:
        from huggingface_hub import hf_hub_download

        # Download the video file from the HuggingFace dataset repository
        local_path = hf_hub_download(repo_id="luciehmct/mammalps-test-recognition", filename=clip_path, repo_type="dataset")

        if os.path.exists(local_path):
            return [local_path]
        else:
            eval_logger.error(f"Downloaded file does not exist: {local_path}")
            return [clip_path]

    except Exception as e:
        eval_logger.error(f"Failed to download video {clip_path}: {str(e)}")
        return [clip_path]


def mammalps_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extract the prompt from the new nested dataset structure.

    Args:
        doc (dict): A dictionary representing the document with nested structure for each subtask.
        lmms_eval_specific_kwargs (dict, optional): Configuration parameters containing:
            - 'subtask': String specifying the task type ('animal', 'action', or 'activity')
            Defaults to None, which results in 'animal' subtask being used.

    Returns:
        str: The prompt for the specified mammalps subtask.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    subtask = lmms_eval_specific_kwargs.get("subtask", "animal")

    # DEBUG: Log what subtask we are generating prompt for with clip and question info
    eval_logger.debug(f"doc_to_text called with subtask: {subtask}")
    eval_logger.debug(f"doc_to_text kwargs: {lmms_eval_specific_kwargs}")
    eval_logger.debug(f"doc_to_text - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"doc_to_text - question_id: {doc.get('id', 'Unknown')}")

    # Use the detected subtask
    eval_logger.debug(f"Final determined subtask for prompt: {subtask}")

    # Extract prompt from the nested structure
    if subtask in doc and "prompt" in doc[subtask]:
        prompt = doc[subtask]["prompt"]
        eval_logger.debug(f"Generated {subtask} prompt with <video> placeholder - length: {len(prompt)} chars")
        eval_logger.debug(f"FULL {subtask.upper()} PROMPT:\n{prompt}")
        
        return prompt
    else:
        eval_logger.error(f"No prompt found for subtask '{subtask}' in document")
        return ""


def mammalps_doc_to_target(doc, lmms_eval_specific_kwargs=None):
    """Extract target answer from the new nested dataset structure.

    Args:
        doc (dict): A dictionary representing the document with nested structure for each subtask.
        lmms_eval_specific_kwargs (dict, optional): Configuration parameters containing:
            - 'subtask': String specifying the task type ('animal', 'action', or 'activity')
            Defaults to None, which results in 'animal' subtask being used.

    Returns:
        list: The target answer for the specified mammalps subtask. Returns:
            - Animal target if subtask is 'animal'
            - Action target if subtask is 'action'
            - Activity target if subtask is 'activity'
            - Empty list if subtask is not recognized or no target found
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    subtask = lmms_eval_specific_kwargs.get("subtask", "animal")

    # Detect the subtask from the call stack
    # since lmms_eval_specific_kwargs may not contain the subtask
    import inspect

    detected_subtask = subtask

    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals

            # Look for task-related variables in the call stack
            if "task" in frame_locals:
                task_obj = frame_locals["task"]
                if hasattr(task_obj, "_config") and hasattr(task_obj._config, "task"):
                    task_name = task_obj._config.task
                    eval_logger.debug(f"Found task name: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"
                    break

            # Additional check for task name in frame
            if "self" in frame_locals and hasattr(frame_locals["self"], "_config"):
                config = frame_locals["self"]._config
                if hasattr(config, "task"):
                    task_name = config.task
                    eval_logger.debug(f"Found task name string: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"

            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"Warning: Could not extract task from stack in doc_to_target: {e}")

    eval_logger.debug(f"Final determined subtask: {detected_subtask}")

    # DEBUG: Log what subtask we are extracting target for
    eval_logger.debug(f"doc_to_target called with subtask: {detected_subtask}")
    eval_logger.debug(f"doc_to_target - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"doc_to_target - question_id: {doc.get('id', 'Unknown')}")

    # Extract target from the nested structure
    if detected_subtask in doc and "answer" in doc[detected_subtask]:
        target = doc[detected_subtask]["answer"]
        eval_logger.debug(f"Returning {detected_subtask} target from nested structure: {target}")
        return target
    else:
        eval_logger.error(f"No answer found for subtask '{detected_subtask}' in document")
        return []


def mammalps_jaccard_metric(predictions, references):
    """Calculate the Jaccard similarity between predicted and reference labels.

    Args:
        predictions (list): A list of predicted labels.
        references (list): A list of reference labels.

    Returns:
        dict: A dictionary containing the Jaccard similarity score.
    """
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
                    eval_logger.debug(f"Extracted via final_answer format: {predicted_labels}")
            except Exception as e:
                eval_logger.warning(f"Failed to eval final_answer match: {e}")

        # Fallback: try to extract any list-like structure
        if not predicted_labels:
            list_match = re.search(r"\[([^\]]+)\]", response)
            if list_match:
                content = list_match.group(1)
                # Split by comma and clean up
                predicted_labels = [item.strip().strip("'\"") for item in content.split(",")]
                eval_logger.debug(f"Fallback : Extracted via list format: {predicted_labels}")

        if not predicted_labels:
            eval_logger.warning("No labels extracted from response! Using empty list.")
            predicted_labels = []

        parsed_prediction = predicted_labels
    else:
        # If predictions is already a list, use it directly
        parsed_prediction = predictions if isinstance(predictions, list) else [predictions]

    # Calculate jaccard index
    jaccard_score = jaccard_index(parsed_prediction, references)
    eval_logger.debug(f"Jaccard calculation: pred={parsed_prediction}, ref={references}, score={jaccard_score}")

    return {"mammalps_jaccard": jaccard_score}


def mammalps_process_results(doc, results, lmms_eval_specific_kwargs=None):
    """Process model results to extract lists from 'Final answer:' format.

    Args:
        doc (dict): A dictionary representing the document, expected to contain a "conversations" key with a list of messages.
        results (list): A list of model response strings to process.
        lmms_eval_specific_kwargs (dict, optional): Configuration parameters containing:
            - 'subtask': String specifying the task type ('animal', 'action', or 'activity')
            Defaults to None, which results in 'animal' subtask being used.

    Returns:
        list: A list of processed results, each being a dictionary with relevant information.
    """

    if not results or len(results) == 0:
        eval_logger.warning("No results provided to process_results")
        return []

    response = results[0].strip()
    subtask = lmms_eval_specific_kwargs.get("subtask", "animal") if lmms_eval_specific_kwargs else "animal"

    # DEBUG: Log clip_path and question_id for extraction from logs
    eval_logger.debug(f"process_results - clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"process_results - full_model_response: {repr(response)}")

    # Try to detect the subtask from the call stack
    import inspect

    detected_subtask = subtask

    try:
        frame = inspect.currentframe()
        while frame:
            frame_locals = frame.f_locals

            # Look for task-related variables in the call stack
            if "task" in frame_locals:
                task_obj = frame_locals["task"]
                if hasattr(task_obj, "_config") and hasattr(task_obj._config, "task"):
                    task_name = task_obj._config.task
                    eval_logger.debug(f"Found task name in process_results: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"
                    break

            # Additional check for task name in frame
            if "self" in frame_locals and hasattr(frame_locals["self"], "_config"):
                config = frame_locals["self"]._config
                if hasattr(config, "task"):
                    task_name = config.task
                    eval_logger.debug(f"Found task name string in process_results: {task_name}")
                    if "action" in task_name:
                        detected_subtask = "action"
                    elif "activity" in task_name:
                        detected_subtask = "activity"
                    elif "animal" in task_name:
                        detected_subtask = "animal"

            frame = frame.f_back
    except Exception as e:
        eval_logger.warning(f"Warning: Could not extract task from stack in process_results: {e}")

    # Use the detected subtask
    subtask = detected_subtask

    # DEBUG: Log the actual model response with clip and question info
    eval_logger.debug(f"Processing model response for subtask: {subtask}")
    eval_logger.debug(f"clip_path: {doc.get('clip', 'Unknown')}")
    eval_logger.debug(f"question_id: {doc.get('id', 'Unknown')}")
    eval_logger.debug(f"Full response: {repr(response)}")

    # Extract list from response - look for "Final answer:" pattern first
    predicted_labels = []

    # Try to extract from "Final answer: ['item1', 'item2']" format
    final_answer_match = re.search(r"Final answer:\s*(\[.*?\])", response, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        try:
            predicted_labels = eval(final_answer_match.group(1))
            if isinstance(predicted_labels, list):
                predicted_labels = [str(label).strip() for label in predicted_labels]
                eval_logger.debug(f"Extracted via final_answer format: {predicted_labels}")
        except Exception as e:
            eval_logger.warning(f"Warning: Failed to eval final_answer match: {e}")

    # Fallback: try to extract any list-like structure
    if not predicted_labels:
        list_match = re.search(r"\[([^\]]+)\]", response)
        if list_match:
            content = list_match.group(1)
            # Split by comma and clean up
            predicted_labels = [item.strip().strip("'\"") for item in content.split(",")]
            eval_logger.debug(f"Fallback : Extracted via list format: {predicted_labels}")

    # Final fallback: try to extract individual quoted items
    if not predicted_labels:
        quoted_items = re.findall(r"['\"]([^'\"]+)['\"]", response)
        if quoted_items:
            predicted_labels = quoted_items
            eval_logger.debug(f"Fallback : Extracted via quoted format: {predicted_labels}")

    if not predicted_labels:
        eval_logger.warning("Warning: No labels extracted from response! Returning empty list.")
        predicted_labels = []

    eval_logger.debug(f"Final extracted labels: {predicted_labels}")

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

    # Get the target for comparison using nested structure
    target_labels = []
    if subtask in doc and "answer" in doc[subtask]:
        target_labels = doc[subtask]["answer"]
        eval_logger.debug(f"Extracted target from nested structure: {target_labels}")
    else:
        eval_logger.error(f"No answer found for subtask '{subtask}' in document structure")
        target_labels = []
    
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Calculate jaccard score
    jaccard_score = jaccard_index(predicted_labels, target_labels)
    eval_logger.info(f"INFO: Jaccard score: pred={predicted_labels}, target={target_labels}, score={jaccard_score}")
    eval_logger.debug(f"INFO: Final result for clip_path={doc.get('clip', 'Unknown')}, question_id={doc.get('id', 'Unknown')}, subtask={subtask}")

    # === Save comprehensive results to subtask-specific JSONL files ===
    try:
        doc_id = doc.get("id", "Unknown")
        clip_path = doc.get("clip", "Unknown")
        
        # Get the enhanced prompt with frame timestamps from InternVL3's cache
        original_prompt = ""
        enhanced_prompt = ""
        try:
            # InternVL3 saves enhanced prompts to mammalps_enhanced_prompts.json
            prompt_cache_file = "mammalps_enhanced_prompts.json"
            if os.path.exists(prompt_cache_file):
                with open(prompt_cache_file, "r", encoding="utf-8") as f:
                    prompt_cache = json.load(f)
                
                doc_id_str = str(doc.get("id", "Unknown"))
                
                if doc_id_str in prompt_cache:
                    # Use the enhanced prompt with frame timestamps
                    enhanced_prompt = prompt_cache[doc_id_str].get("enhanced_prompt_with_timestamps", "")
                    original_prompt = prompt_cache[doc_id_str].get("original_prompt", "")
                    eval_logger.debug(f"Retrieved enhanced prompt with frame timestamps from InternVL3 cache for doc_id: {doc_id_str}")
                else:
                    eval_logger.debug(f"No cached enhanced prompt found for doc_id: {doc_id_str}")
            else:
                eval_logger.debug(f"Enhanced prompt cache file not found: {prompt_cache_file}")
        except Exception as e:
            eval_logger.debug(f"Could not load InternVL3 enhanced prompt cache: {e}")
        
        # Fallback to original prompt from document if no enhanced prompt available
        if not enhanced_prompt and subtask in doc and "prompt" in doc[subtask]:
            original_prompt = doc[subtask]["prompt"]
            enhanced_prompt = original_prompt
            eval_logger.debug(f"Using fallback prompt from document for subtask {subtask}")
        elif not enhanced_prompt:
            eval_logger.debug(f"Could not find any prompt for subtask {subtask}")
        
        # Create comprehensive result entry - with jaccard_score included
        comprehensive_result = {
            "id": doc_id,
            "clip": clip_path,
            "prompt": enhanced_prompt,  # Use enhanced prompt with frame timestamps
            "full_answer": response,
            "answer": predicted_labels,
            "ground_truth": target_labels,
            "jaccard_score": jaccard_score
        }
        
        # Create model_used_date_time directory structure
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_name = "InternVL3-8B"  # Can be made configurable if needed
        output_dir = f"{model_name}_{timestamp}"
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to subtask-specific JSONL file in the structured directory
        subtask_file = os.path.join(output_dir, f"mammalps_{subtask}.jsonl")
        
        # Append to the subtask-specific file
        with open(subtask_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(comprehensive_result, ensure_ascii=False) + "\n")
            
        eval_logger.debug(f"Saved comprehensive result to {subtask_file}: {doc_id}")
        
    except Exception as e:
        eval_logger.warning(f"Failed to save comprehensive result: {e}")

    # Return the computed jaccard score
    return {"jaccard": jaccard_score}


# Aggregation function for mammalps jaccard results
def mammalps_jaccard_aggregation(items):
    """Aggregate Jaccard scores by averaging them.

    Args:
        items (list): A list of Jaccard score dictionaries from individual evaluations.

    Returns:
        float: The average Jaccard score.
    """
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
            eval_logger.warning(f"Warning: Aggregation received raw list: {item}")
        else:
            eval_logger.warning(f"Warning: Unexpected item type in aggregation: {type(item)}, value: {item}")

    return total_jaccard / count if count > 0 else 0.0
