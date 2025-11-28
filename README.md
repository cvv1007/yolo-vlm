1. **Dataset**

   Create a folder named `CODAdatasets` in the project root.  
   Download the two preprocessed datasets **CODA2022-val** and **CODA2022-test** from:

   https://drive.google.com/drive/folders/1Kbb1sPsmZd9K2N8P0BeBjPHqPhuso6IN?usp=share_link

   and put them into the `CODAdatasets` folder.

2. **How to Run the Project**

   There are detailed instructions in each script file. A typical workflow is:

   - Ensure all the dependencies are installed.
   - Run `scripts/build_gallery.py` to generate the data gallery using **CODA2022-val**.  
     Check and ensure that you get all the categorized data for all categories mentioned in `annotations.json` in CODA2022-val.
   - Run `scripts/build_faiss_index.py` to generate the FAISS index database.  
     This is the foundation of our classification/retrieval pipeline.
   - Run `end2end.py` to perform object detection on **CODA2022-test**.
   - Evaluate the results by running `evaluate/evaluate.py`.

3. **Project Structure**

```text
YOLOPRACTICE/
├── CODAdatasets/                      # Dataset directory
│
├── data_gallery/                      # Gallery data for retrieval or evaluation
│
├── data_preprocess/                   # Data preprocessing scripts
│
├── detector/
│   └── yolo_crop.py                   # YOLO-based detector & cropper
│
├── eval_test/                         # Score results for detection output
│
├── evaluate/                          # Evaluation module
│   ├── data/                          # Ground-truth data for evaluation
│   │
│   ├── evaluate_corner_cat.py         # Extra corner-point evaluation script (optional)
│   ├── evaluate.py                    # Main evaluation script
│   ├── extract_cat99.py               # Script assisting manual checking of objects recognized by Qwen
│   ├── fix_cat_99.py                  # Script assisting manual checking of objects recognized by Qwen
│   └── generate_gt.py                 # Generate ground-truth annotation from original annotation
│
├── output/                            # Model prediction outputs
│
├── retriever/                         # Feature retrieval module
│   ├── coreSearcher.py                # Core retrieval logic
│   └── utils.py                       # Qwen utility functions
│
├── scripts/                           # Build the basic query database
│   ├── build_faiss_index.py           # Build FAISS index
│   ├── build_gallery.py               # Build image gallery
│
├── submission/                        # Generated prediction submissions
│
├── weights/                           # Model weights (original YOLO and our fine-tuned YOLO11m)
│
└── end2end.py                         # End-to-end pipeline combining YOLO detection, FAISS retrieval,
                                       # and Qwen-based disambiguation for uncertain cases
