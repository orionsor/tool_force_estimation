# Grasp Independent Indirect Tool Force Estimation using Vision based Tactile Sensors

This repository provides the implementation for **Indirect Tool Force Estimation using Vision based Tactile Sensors** .  
It supports both **normal force** and **shear force** estimation using vision-based tactile sensors.  

---
## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/orionsor/tool_force_estimation.git
cd tool_force_estimation

### 2.Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

## 📂 Dataset Structure

The dataset contains the original tactile images(from sensors on two fingers) and their corresponding ground truth force data. the structure is as follows:

data_normal/
│
├── trial1/
│ ├── tactile1/
│ │ ├── tactile_frame_0000.jpg
│ │ ├── tactile_frame_0001.jpg
│ │ └── ...
│ ├── tactile2/
│ │ ├── tactile_frame_0000.jpg
│ │ ├── tactile_frame_0001.jpg
│ │ └── ...
│ └── force/
│ ├── force_data_0000.txt
│ ├── force_data_0001.txt
│ └── ...
│
├── trial2/
│ ├── tactile1/
│ ├── tactile2/
│ └── force/
│
└── ...
## unzip the dataset
```bash
unzip data_normal.zip -d ./data/normal
unzip data_shear.zip -d ./data/shear
```

### 3. Run training / evaluation

## Normal force estimation
```bash
python main_normal.py
```

## Shear force estimation
```bash
python main_shear.py
```






