import re
import os
import json

from loguru import logger as eval_logger


def extract_final_answer(response, subtask="animal", artifacts_to_filter=None):
    """Enhanced function to extract final answers from model responses with robust parsing.
    
    Handles various edge cases including:
    - LaTeX formatting (\\boxed{...})
    - Escaped characters
    - Implicit answers without "Final answer:" prefix
    - Multiple bracket formats
    - Malformed responses
    
    Args:
        response (str): The model response to parse
        subtask (str): The task type ('animal', 'action', 'activity') for context-specific filtering
        artifacts_to_filter (list): Additional artifacts to filter out
        
    Returns:
        list: Extracted and cleaned labels
    """
    if not isinstance(response, str) or not response.strip():
        eval_logger.warning("Empty or invalid response provided to extract_final_answer")
        return []
    
    # Default artifacts to filter out
    default_artifacts = [
        "s execute", "```python", "recognize(", "entity_type", "def ", "return", "import", "from ", "print(", "Example:",
        "python", "code", "function", "list", "str", "List", "str", "Optional", "execute", "```", "final answer",
        # Filter out entity type parameters that are not actual answers
        "animal", "action", "activity"
    ]
    
    # Add animal names to filter for activity tasks
    if subtask == "activity":
        default_artifacts.extend([
            "deer", "elk", "moose", "bear", "wolf", "fox", "rabbit", "squirrel", "bird", "cat", "dog",
            "red_deer", "roe_deer", "wild_boar", "lynx", "badger", "otter", "marten", "weasel"
        ])
    
    # Combine with any additional artifacts
    all_artifacts = default_artifacts + (artifacts_to_filter or [])
    
    def clean_item(item):
        """Clean and validate a single extracted item"""
        if not item:
            return None
            
        # Remove leading/trailing whitespace and quotes
        item = item.strip().strip('"\'')
        
        # Handle LaTeX boxed formatting
        latex_match = re.search(r'\\boxed\{([^}]+)\}', item)
        if latex_match:
            item = latex_match.group(1).strip()
        
        # Remove any remaining LaTeX artifacts
        item = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', item)  # Remove \command{content} -> content
        item = re.sub(r'\\[a-zA-Z]+', '', item)  # Remove standalone \commands
        
        # Handle escaped characters
        item = item.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')
        
        # Remove leading/trailing punctuation and colons
        item = re.sub(r'^[:\s,;.-]+', '', item)
        item = re.sub(r'[:\s,;.-]+$', '', item)
        
        # Final cleanup
        item = item.strip().strip('"\'')
        
        # Validate item
        if (item and len(item) > 1 and 
            not any(artifact.lower() in item.lower() for artifact in all_artifacts) and
            not item.endswith("(") and not item.startswith("(") and
            not re.match(r'^[^a-zA-Z]*$', item) and  # Not just symbols/numbers
            len(item) < 100):  # Reasonable length limit
            return item.lower()  # Normalize to lowercase
        return None
    
    predicted_labels = []
    
    eval_logger.debug(f"extract_final_answer input: {repr(response[:200])}")
    
    # Strategy 1: Look for "Final answer:" with various bracket formats
    final_answer_patterns = [
        r"final answer:\s*\[([^\]]*)\]",  # Final answer: [...]
        r"final answer:\s*\\boxed\{([^}]*)\}",  # Final answer: \boxed{...}
        r"final answer:\s*\{([^}]*)\}",  # Final answer: {...}
        r"final answer:\s*\(([^)]*)\)",  # Final answer: (...)
        r"final answer[:\s]*([^\n]*)",   # Final answer: anything till newline
    ]
    
    for pattern in final_answer_patterns:
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
        for match in matches:
            content = match.group(1).strip()
            if content:
                # Handle list-like content
                if any(sep in content for sep in [',', ';', '|']):
                    # Split by common separators
                    items = re.split(r'[,;|]', content)
                    for item in items:
                        cleaned = clean_item(item)
                        if cleaned and cleaned not in predicted_labels:
                            predicted_labels.append(cleaned)
                else:
                    # Single item
                    cleaned = clean_item(content)
                    if cleaned and cleaned not in predicted_labels:
                        predicted_labels.append(cleaned)
                
                if predicted_labels:
                    eval_logger.debug(f"Extracted via final_answer pattern '{pattern}': {predicted_labels}")
                    break
        if predicted_labels:
            break
    
    # Strategy 2: Look for bracket structures without "Final answer:" prefix
    if not predicted_labels:
        bracket_patterns = [
            r'\[([^\]]+)\]',  # [...]
            r'\\boxed\{([^}]+)\}',  # \boxed{...}
            r'\{([^}]+)\}',  # {...}
        ]
        
        for pattern in bracket_patterns:
            matches = re.finditer(pattern, response)
            for match in matches:
                content = match.group(1).strip()
                if content and ('"' in content or "'" in content or ',' in content):
                    # Only process if it looks like a list
                    items = re.split(r'[,;|]', content)
                    temp_labels = []
                    for item in items:
                        cleaned = clean_item(item)
                        if cleaned and cleaned not in temp_labels:
                            temp_labels.append(cleaned)
                    
                    if temp_labels:
                        predicted_labels = temp_labels
                        eval_logger.debug(f"Extracted via bracket pattern '{pattern}': {predicted_labels}")
                        break
            if predicted_labels:
                break
    
    # Strategy 3: Look for quoted items (more conservative fallback)
    if not predicted_labels:
        # Only if response doesn't seem incomplete
        response_seems_incomplete = (
            response.strip().endswith("```") or 
            response.strip().endswith("recognize('animal')") or
            response.strip().endswith("recognize('action')") or
            response.strip().endswith("recognize('activity')") or
            len(response.strip()) < 20
        )
        
        if not response_seems_incomplete:
            quoted_items = re.findall(r'["\']([^"\']{2,50})["\']', response)
            for item in quoted_items:
                cleaned = clean_item(item)
                if cleaned and cleaned not in predicted_labels:
                    predicted_labels.append(cleaned)
            
            if predicted_labels:
                eval_logger.debug(f"Extracted via quoted items: {predicted_labels}")
    
    # Strategy 4: Look for implicit answers (lines that look like answers)
    if not predicted_labels:
        # Look for lines that might be implicit answers
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Skip obvious non-answer lines
            if (line and not line.startswith(('#', '//', '/*', 'def ', 'import', 'from ')) and
                not any(artifact.lower() in line.lower() for artifact in default_artifacts[:10]) and
                len(line) > 2 and len(line) < 50):
                
                # Check if line contains potential answers
                if any(sep in line for sep in [',', ';', 'and ', 'or ']):
                    # Split potential multi-item lines
                    items = re.split(r'[,;]|\s+and\s+|\s+or\s+', line)
                    for item in items:
                        cleaned = clean_item(item)
                        if cleaned and cleaned not in predicted_labels:
                            predicted_labels.append(cleaned)
                else:
                    # Single potential answer
                    cleaned = clean_item(line)
                    if cleaned and cleaned not in predicted_labels:
                        predicted_labels.append(cleaned)
        
        if predicted_labels:
            eval_logger.debug(f"Extracted via implicit answers: {predicted_labels}")
    
    # Remove duplicates while preserving order
    unique_labels = []
    seen = set()
    for label in predicted_labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)
    
    eval_logger.debug(f"extract_final_answer result: {unique_labels}")
    return unique_labels


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

        # Download the video file from the updated HuggingFace dataset repository
        local_path = hf_hub_download(repo_id="luciehmct/mammalps", filename=clip_path, repo_type="dataset")

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
        
        # Store the original prompt for later retrieval with timestamps
        doc_id = doc.get("id", "Unknown")
        try:
            # Create a simple cache for the original prompt to be used by process_results
            original_prompt_cache_file = "mammalps_original_prompts.json"
            
            # Load existing cache or create new
            try:
                with open(original_prompt_cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                cache = {}
            
            # Store the original prompt with subtask info
            cache_key = f"{doc_id}_{subtask}"
            cache[cache_key] = {
                "id": doc_id,
                "subtask": subtask,
                "original_prompt": prompt,
                "clip_path": doc.get("clip", "Unknown")
            }
            
            # Save updated cache
            with open(original_prompt_cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            eval_logger.debug(f"Failed to save original prompt: {e}")
        
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
            - Answer list if found in nested structure
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

    # Extract answer from the nested structure
    if detected_subtask in doc and "answer" in doc[detected_subtask]:
        target = doc[detected_subtask]["answer"]
        eval_logger.debug(f"Returning {detected_subtask} target: {target}")
        return target
    else:
        eval_logger.warning(f"No answer found for subtask '{detected_subtask}' in document")
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
        # Use the enhanced extract_final_answer function
        predicted_labels = extract_final_answer(response)
        eval_logger.debug(f"mammalps_jaccard_metric extracted: {predicted_labels}")
    else:
        # If predictions is already a list, use it directly
        predicted_labels = predictions if isinstance(predictions, list) else [predictions]

    # Calculate jaccard index
    jaccard_score = jaccard_index(predicted_labels, references)
    eval_logger.debug(f"Jaccard calculation: pred={predicted_labels}, ref={references}, score={jaccard_score}")

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
    eval_logger.debug(f"process_results - question_id: {doc.get('id', 'Unknown')}")
    eval_logger.debug(f"process_results - subtask: {subtask}")
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

    # Extract list from response using enhanced parsing function
    predicted_labels = extract_final_answer(response, subtask)
    
    eval_logger.debug(f"Raw response for parsing: {repr(response[:500])}")  # Log first 500 chars for debugging
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

    # Get the target for comparison from the nested document structure
    target_labels = []
    
    # The document has nested structure: doc[subtask]["answer"]
    if subtask in doc and isinstance(doc[subtask], dict) and "answer" in doc[subtask]:
        target_labels = doc[subtask]["answer"]
        eval_logger.debug(f"Found target in nested structure: {target_labels}")
    else:
        # Fallback to direct access
        target_labels = doc.get(subtask, [])
        if not target_labels:
            # Try alternative key names
            target_labels = doc.get("answer", [])
        eval_logger.debug(f"Found target via fallback: {target_labels}")
    
    # Ensure target_labels is a list
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    elif not isinstance(target_labels, list):
        target_labels = []

    # Calculate jaccard score
    jaccard_score = jaccard_index(predicted_labels, target_labels)
    eval_logger.info(f"INFO: Jaccard score: pred={predicted_labels}, target={target_labels}, score={jaccard_score}")
    eval_logger.debug(f"INFO: Final result for clip_path={doc.get('clip', 'Unknown')}, question_id={doc.get('id', 'Unknown')}, subtask={subtask}")

    # Save detailed results to a custom JSONL file
    try:
        # Try to get full prompt with timestamps from cache
        doc_id_key = doc.get("id", "Unknown")
        full_prompt_with_timestamps = "N/A - Not available"
        
        try:
            # Load timestamped prompt cache
            with open("mammalps_prompt_cache.json", "r", encoding="utf-8") as f:
                prompt_cache = json.load(f)
                
            # Try both string and integer keys for the doc_id with subtask
            cached_info = None
            cache_key_with_subtask = f"{doc_id_key}_{subtask}"
            
            # Try subtask-specific key first
            if cache_key_with_subtask in prompt_cache:
                cached_info = prompt_cache[cache_key_with_subtask]
                eval_logger.debug(f"Retrieved prompt from cache for ID {cache_key_with_subtask}")
            elif str(doc_id_key) in prompt_cache:
                cached_info = prompt_cache[str(doc_id_key)]
                eval_logger.debug(f"Retrieved prompt from cache for ID {doc_id_key} (string key)")
            elif doc_id_key in prompt_cache:
                cached_info = prompt_cache[doc_id_key]
                eval_logger.debug(f"Retrieved prompt from cache for ID {doc_id_key} (direct key)")
            else:
                eval_logger.debug(f"ID {doc_id_key} not found in prompt cache (available keys: {list(prompt_cache.keys())})")
                
            if cached_info:
                full_prompt_with_timestamps = cached_info.get("full_prompt_with_timestamps", "N/A")
                
        except Exception as e:
            eval_logger.debug(f"Could not load prompt cache: {e}")
            
        # If we couldn't get the timestamped prompt, try to fallback to original prompt
        if full_prompt_with_timestamps == "N/A - Not available":
            try:
                # Load original prompt cache as fallback
                with open("mammalps_original_prompts.json", "r", encoding="utf-8") as f:
                    original_cache = json.load(f)
                    
                cache_key = f"{doc_id_key}_{subtask}"
                if cache_key in original_cache:
                    original_info = original_cache[cache_key]
                    full_prompt_with_timestamps = original_info.get("original_prompt", "N/A")
                    eval_logger.debug(f"Used original prompt as fallback for ID {doc_id_key}")
                    
            except Exception as e:
                eval_logger.debug(f"Could not load original prompt cache: {e}")
        
        # Extract the clean ground truth for output
        clean_ground_truth = target_labels  # We already processed target_labels above
            
        # Create the output entry with the exact format you requested
        output_entry = {
            "id": doc_id_key,
            "clip_path": doc.get("clip", "Unknown"), 
            "prompt": full_prompt_with_timestamps,
            "full_answer": response,
            "answer": predicted_labels,
            "ground_truth": clean_ground_truth,
            "jaccard_score": jaccard_score
        }
        
        # Create folder structure: Model_date_hour/mammalps_task.jsonl
        from datetime import datetime
        import os
        current_time = datetime.now()
        date_hour = current_time.strftime("%Y%m%d_%H%M")
        
        # Extract model name from the call stack or use default
        model_name = "InternVL3"  # Default model name
        try:
            import inspect
            frame = inspect.currentframe()
            while frame:
                frame_locals = frame.f_locals
                # Look for model information in the call stack
                if "self" in frame_locals and hasattr(frame_locals["self"], "path"):
                    model_path = frame_locals["self"].path
                    if "InternVL3" in model_path:
                        model_name = "InternVL3"
                    elif "InternVL" in model_path:
                        model_name = "InternVL"
                    elif "gpt" in model_path.lower():
                        model_name = "GPT"
                    else:
                        # Extract model name from path
                        if "/" in model_path:
                            model_name = model_path.split("/")[-1]
                        else:
                            model_name = model_path
                    break
                frame = frame.f_back
        except Exception as e:
            eval_logger.debug(f"Could not extract model name: {e}")
        
        # Create folder structure in results directory
        results_dir = "results"
        folder_name = f"{model_name}_{date_hour}"
        full_folder_path = os.path.join(results_dir, folder_name)
        os.makedirs(full_folder_path, exist_ok=True)
        
        # Create filename: results/model_date_hour/mammalps_task.jsonl
        output_filename = os.path.join(full_folder_path, f"mammalps_{subtask}.jsonl")
        
        # Append to file (create if doesn't exist)
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            
        eval_logger.debug(f"Saved detailed results to {output_filename}")
        
    except Exception as e:
        eval_logger.warning(f"Warning: Failed to save detailed results: {e}")

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
