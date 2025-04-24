# Objects365 Dataset Downloader & Converter


This repo holds **Zero dependencies**  script to download  based upon 

This repository provides a robust Python script for downloading, extracting, and converting the [`Object365`](https://www.objects365.org/download.html) dataset to YOLO format. The implementation includes significant improvements over the original [Ultralytics](https://github.com/ultralytics) [script](https://github.com/ultralytics/yolov5/blob/master/data/Objects365.yaml#L402).

## Features
- ✅ Resumable downloads: Can continue interrupted downloads from where they left off
- ✅ Parallel processing: Multi-threaded downloading and extraction for improved performance
- ✅ Smart extraction: Multiple archive formats supported (.zip, .tar, .gz)
- ✅ Automatic YOLO conversion: Converts COCO-format annotations to YOLO format
- ✅ Progress tracking: Visual progress bars for downloads and processing
- ✅ Fault tolerance: Uses marker files to track completed downloads/extractions for robust operation
- ✅ Resource efficient: Optimized for handling the large dataset (1.7M+ images)

## Dependencies
- Python 3.6+
- pycocotools>=2.0
- requests
- tqdm
- numpy

## Usage
Simple usage:

```bash
python download.py
```

The script will:

Download the Objects365 dataset (both training and validation sets)
Extract all archives
Convert annotations to YOLO format
Organize files in the appropriate directory structure
By default, the script uses 16 threads for parallel downloading. You can modify this in the script.

## Output Structure
```
/path/to/output_dir/
├── images/
│   ├── train/      # ~1.7M training images
│   └── val/        # ~80K validation images
├── labels/
│   ├── train/      # YOLO format annotations for training
│   └── val/        # YOLO format annotations for validation
└── zhiyuan_objv2_*.json  # Original annotation files
```

## Configuration
Edit the [base_dir](https://github.com/Simo93-rgb/Object365-download/blob/main/download.py#L290) variable in the script to change the output directory:
```python
base_dir = Path("/your/preferred/path")
```
## Licence
The Objects365 dataset is available for academic purposes only. Please read the license in the [official documentation](https://www.objects365.org/download.html).

## Citation
If you use the Objects365 dataset in your research, please cite:
```citation
@inproceedings{shao2019objects365,
    title={Objects365: A Large-Scale, High-Quality Dataset for Object Detection},
    author={Shao, Shuai and Li, Zeming and Zhang, Tianyuan and Peng, Chao and Yu, Gang and Zhang, Xiangyu and Li, Jing and Sun, Jian},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={8430--8439},
    year={2019}
}
```
















