Aimonk Multilabel Image Classification Assignment 

Overview 

This project addresses a multi-label image classification problem where each image 
may contain multiple attributes simultaneously. The dataset consists of images of 
clothing items along with annotations for four attributes per image. 

Key challenges of the dataset include: 
        Presence of multiple labels per image 
        Missing attribute information marked as “NA” 
        Significant class imbalance 
        Real-world variability in image content (pose, background, framing) 
        The objective is to train a deep learning model that can predict all attributes present in a 
        given image. 

Problem Formulation 
        This is a multi-label classification task 
        For each image: 
                Each attribute is independent 
                Multiple attributes may be present simultaneously 
                Absence of an attribute is explicitly labelled as “0” 
                Missing information is labelled as “NA” 

Methodology

1. Model Architecture 
        A pretrained ResNet50 model (trained on ImageNet) was used as the backbone. 
        Reasons for this choice: 
                Strong feature extraction capability 
                Proven performance on visual recognition tasks 
                Suitable for fine-tuning on moderate-sized datasets 
                EAicient trade-oA between accuracy and computational cost 
                The final fully connected layer was replaced with a layer producing 4 outputs (one per                       
                attribute). 
                Sigmoid activation is applied during inference to obtain independent probabilities for       
                each attribute. 

2. Transfer Learning Strategy 
        The model was initialized with ImageNet-pretrained weights and fine-tuned on the         
        provided dataset. 
        Training from scratch was avoided because: 
                Dataset size is limited 
                Transfer learning improves convergence speed 
                Pretrained features generalize well to clothing attributes

3. Handling Missing Labels (NA) 
        The dataset contains attributes marked as “NA”, indicating that the attribute   
        information is unavailable rather than absent. 
        To address this: 
                A mask tensor was created for each sample 
                Positions corresponding to NA values were excluded from loss computation 
                Images with partial labels were still used for training 
                This prevents incorrect supervision while maximizing data utilization. 

4. Loss Function 
        A Masked Binary Cross-Entropy Loss with Logits was used. 
        Key properties: 
                Supports multi-label classification 
                Computes independent loss per attribute 
                Ignores NA positions via masking 
                Allows all usable annotations to contribute to training 
        
5. Handling Class Imbalance 
        The dataset exhibits skewed distribution of attributes. 
        To mitigate imbalance effects: 
                Binary Cross-Entropy loss was used (robust baseline) 
                Mini-batch shuAling applied 
                Data augmentation used to improve generalization 
        Potential improvements (not implemented due to time constraints): 
                Class-weighted loss 
                Focal loss 
                Oversampling of minority attributes 

6. Data Preprocessing and Augmentation 
        Images were resized to 224×224 pixels and normalized using ImageNet statistics. 
        Moderate augmentations were applied: 
        - Random horizontal flipping 
        - Colour jitter (brightness, contrast) 
        - Standard normalization 
        - Aggressive geometric augmentations were avoided to preserve clothing semantics. 

7. Training Configuration 
        Optimizer: AdamW 
        Learning rate: 1e-4 
        Batch size: 8 
        Loss: Masked BCEWithLogitsLoss 
        Device: GPU if available 
Training loss was tracked per iteration. 

8. Loss Curve Visualization 
    A loss curve was generated as required: 
        X-axis: iteration_number   
        Y-axis: training_loss   
        Title: Aimonk_multilabel_problem   
    This provides insight into training dynamics. 

9. Inference Procedure 
        For a given input image: 
                1. Image is preprocessed using the same transforms as training 
                2. Model produces raw logits 
                3. Sigmoid activation converts logits to probabilities 
                4. Attributes with probability > 0.5 are considered present 
        Output is a list of predicted attributes. 

Robustness Measures 

To ensure reliable training:
        Missing image files are automatically skipped 
        Corrupted images are handled safely 
        Dataset loading does not crash on invalid samples 
        Partial annotations are fully supported 
        These steps simulate real-world production pipelines.

Potential Improvements 
        If additional time were available, the following techniques could further improve 
        performance: 
        Advanced Modeling - EAicientNet or Vision Transformers - Multi-task learning with separate attribute heads - Attribute correlation modeling 
        Imbalance Handling 
        - Focal loss 
        - Class-balanced loss 
        - Weighted sampling 
        Data-Centric Improvements 
        - CutMix / MixUp 
        - Synthetic augmentation 
        - Hard example mining 

Evaluation Enhancements 
        Per-attribute precision/recall 
        F1 score 
        ROC-AUC per label 
        Calibration of thresholds per attribute 
Files Included 
  dataset.py -Dataset loader with NA handling 
  model.py -ResNet50 model definition 
  loss.py -Masked loss implementation 
  train.py -Training script and loss plotting 
  inference.py -Prediction script 
  weights/ -Saved model weights 
  plots/ -Loss curve image 
 
How to Run 
        Training 
        python train.py 
        python inference.py 
Modify the path in the script as needed. 
 
Conclusion 
        This solution provides a robust and practical approach to multi-label image       
        classification under real-world conditions such as missing annotations and class 
        imbalance. 
        The pipeline emphasizes: 
        Correct problem formulation 
        EAicient transfer learning 
        Robust data handling 
        Clean engineering practices 
        Reproducibility 