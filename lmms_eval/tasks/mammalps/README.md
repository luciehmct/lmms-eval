Explanations for dataset_builder.py (explain better later)
USAGE INSTRUCTIONS:

1. Build all datasets with separate train/test directories:
   python3 dataset_builder.py

2. Build specific dataset:
   python3 dataset_builder.py --dataset animalkingdom

3. Build only test split:
   python3 dataset_builder.py --dataset mammalps --split test

4. Create unified dataset (train + test in one directory):
   python3 dataset_builder.py --dataset animalkingdom --unified

5. Build and upload to HuggingFace:
   python3 dataset_builder.py --dataset mammalps --unified --upload --hf-token YOUR_TOKEN

6. Upload all datasets as unified repositories:
   python3 dataset_builder.py --unified --upload --hf-token YOUR_TOKEN

Command-line arguments:
  --dataset, -d    Dataset to build: animalkingdom, mammalnet, mammalps, or all (default: all)
  --split, -s      Split to build: test, train, or both (default: both)
  --unified, -u    Create unified dataset with both splits in one directory
  --upload         Upload to HuggingFace Hub after creation
  --hf-token       HuggingFace token for upload (required if --upload)
  --hf-username    HuggingFace username (default: luciehmct)
  --private        Create private repository on HuggingFace

Examples:
  # Build only MammalAlps unified dataset
  python3 dataset_builder.py -d mammalps -u
  
  # Build and upload AnimalKingdom as unified dataset
  python3 dataset_builder.py -d animalkingdom -u --upload --hf-token hf_xxx

  # Build test split only for all datasets
  python3 dataset_builder.py -s test

7. Build a custom dataset programmatically:
   from dataset_builder import build_custom_dataset
   
   test_data, train_data = build_custom_dataset(
       name="MyDataset",
       base_dir="MyDataset_videos_annotations",
       tasks=["action", "animal"],
       include_conversations=False,
       output_prefix="mydataset",
       distribute_clips=True
   )

8. Required directory structure:
   {base_dir}/
   ├── clips/                    # Source video files
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   ├── test/
   │   ├── action_recognition_cot.jsonl
   │   ├── animal_recognition_cot.jsonl
   │   └── ...
   └── train/
       ├── action_recognition_cot.jsonl
       ├── animal_recognition_cot.jsonl
       └── ...

9. Output structure (separate splits):
   {DatasetName}_HF_Dataset_Test/
   ├── {prefix}_test_dataset.json
   ├── README.md
   └── clips/                    # Video files used in test split
       ├── video1.mp4
       └── ...
   
   {DatasetName}_HF_Dataset_Train/
   ├── {prefix}_train_dataset.json
   ├── README.md
   └── clips/                    # Video files used in train split
       ├── video3.mp4
       └── ...

10. Output structure (unified):
    {DatasetName}_HF_Dataset_Unified/
    ├── README.md
    ├── test/
    │   ├── {prefix}_test_dataset.json
    │   └── clips/
    │       ├── video1.mp4
    │       └── ...
    └── train/
        ├── {prefix}_train_dataset.json
        └── clips/
            ├── video3.mp4
            └── ...

11. Dataset record format:
    {
      "id": original_id,
      "clip": "clips/video.mp4",    # Actual video file path
      "video_id": "video_id",
      "task1": {"prompt": "...", "answer": ["..."]},
      "task2": {"prompt": "...", "answer": ["..."]}
    }
