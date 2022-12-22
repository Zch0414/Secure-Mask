# Secure-Mask
Code for "Self-supervised Multi-Modal Video Forgery Attack Detection", WCNC 2023
Usage for Secure-Mask

## Prepare data

Unfortunately, we cannot publish our dataset. You can collect your own data following Secure-Pose https://ieeexplore.ieee.org/abstract/document/9488798. After you get CSI-Image dataset and the corresponding motion vector, you can follow the following steps to build your own Secure-Mask system.

## Code for Secure-Mask

|-- processing_for_motion_vecor.py
|-- splitting_for_motion_vector_mask.py
|-- main

    |-- models
    
        |-- model_parts.py
        
        |-- models.py
        
    |-- utils
    
        |-- detector_dataset.py |-- segmentor_dataset.py |-- forgery_dataset.py
        
        |-- dice_loss.py
        
    |-- processing_for_attack_detection
    
        |-- data_for_attack_step_1.py |-- data_for_attack_step_2.py

|-- splitting_for_attack.py

|-- detector_trian&eval.py

|-- segmentor_train.py

|-- attack_train&eval.py

## Run

###	Preparation

Run “processing_for_motion_vector.py” can help you generate the pseudo mask from motion vector data.

Run “splitting_for_motion_vector.py” can help you split the motion vector into the training and testing set.

###	Human Detector

Run “detector_train&eval.py” can help you train and evaluate the human detector model.

###	Human Segmentor

Run “segmentor_train.py” and “segmentor_eval.py” can help you train and evaluate the human segmentor.

### Forgery Detector

Run ”data_for_attack_step_1.py” and “data_for_attack_step_2.py” can help you frame-like data for the forgery detector model.

Run “splitting_for_attack.py” can help you generate video-like data from frame-like data. Note that the production of the “splitting_for_attack.py” is not quite compatible with “forgery_dataset.py”. So, you may need to adjust it according to you own dataset. In our case, we need to change the directory name manually.

Run “attack_trian&eval.py” can help you train and evaluate the forgery detector model.

## Acknowledgement

This project is mainly done by Chenhui in his junior year with essential help from Xiang and Rabih. So, the code is not tidy enough and may be difficult to read. We apologize for this. We named our file s\*p\*\_\*\_\*, the first * means the scene number; the second * means the number of people in the scene; the third * means the video number for this situation; the last * means the frame number in this video. We hope the above information can help you understand the code.
