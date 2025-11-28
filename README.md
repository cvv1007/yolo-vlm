1. Dataset: Create the folder named "CODAdatasets" in the project root. Download the two preprocessed dataset: CODA2022-val CODA2022-test from this link: https://drive.google.com/drive/folders/1Kbb1sPsmZd9K2N8P0BeBjPHqPhuso6IN?usp=share_link and put them into "CODAdatasets".
2. Instruction of running the project. There are detailed instructions in each file.
   Ensure all the dependencies installed.
   Run the scripts/build_gallery.py to generate the data gallery using CODA2022-val correctly. Check and ensure that you get all the categoried data of all categories mentioned in annotations.json in CODA2022-val. Then Run the scripts/build_faiss_index.py to generate the faiss index database.This is the foundation of our classification.
   Run the end2end.py to do object detection for CODA2022-test.
   Evaluate by evaluate/evalute.py.
4. YOLOPRACTICE/
├── CODAdatasets/ # Dataset directory
│
├── data_gallery/ # Gallery data for retrieval or evaluation
│
├── data_preprocess/ # Data preprocessing scripts
│
├── detector/
│ └── yolo_crop.py # YOLO-based detector & cropper
│
├── eval_test/ # Score results for detection result
│
├── evaluate/ # Evaluation module
│ ├── data/ # Ground-truth data for evaluation
│ │
│ ├── evaluate_corner_cat.py # extra Corner-point evaluation script(optional)
│ ├── evaluate.py # Main evaluation script
│ ├── extract_cat99.py # Script assisting manually checking objects recognized by Qwen
│ ├── fix_cat_99.py # Script assiting manually checking objects recognized by Qwen
│ └── generate_gt.py # Generate ground-truth annotation from original annotation
│
├── output/ # Model prediction outputs
│
├── retriever/ # Feature retrieval module
│ ├── coreSearcher.py # Core retrieval logic
│ └── utils.py # Qwen utility functions
│
├── scripts/ # build basic query databse
│ ├── build_faiss_index.py # Build FAISS index
│ ├── build_gallery.py # Build image gallery
│
├── submission/ # Generated prediction submissions
│
└── weights/ # Model weights（origin yolo, our fine-tunned yolo11m weights here）
│
└── end2end.py # End-to-end pipeline combining YOLO detection, FAISS retrieval,and Qwen-based disambiguation for uncertain cases



