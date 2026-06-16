# EfficientDet-D0 Object Detection on SH17

This project experiments with **EfficientDet-D0** on the **SH17 object detection dataset**. Link dataset : https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
The goal is to investigate whether a lightweight EfficientDet model can be adapted to a PPE and human-safety detection task.

The experiment does not aim to directly reproduce the original YOLO-based SH17 benchmark. Instead, it focuses on evaluating how different EfficientDet-D0 training strategies affect detection performance, especially under class imbalance and small-object detection challenges.

---

## 1. Project Overview

The SH17 dataset contains **17 object categories**, including human body parts and PPE-related objects. Examples include:

- person
- head
- face
- hands
- ear
- helmet
- safety vest
- medical suit
- earmuffs
- face guard
- gloves
- glasses

The dataset presents two main challenges:

1. **Class imbalance**  
   Common classes such as person, head, face, hands, and ear appear much more frequently than PPE-related classes such as helmet, safety vest, medical suit, earmuffs, and face guard.

2. **Small-object detection**  
   Many PPE objects occupy only a small region in the image. After resizing images to a fixed input size, objects such as gloves, glasses, helmets, earmuffs, and face guards become harder to localize.

Because of these challenges, this project compares several EfficientDet-D0 training variants instead of using only one baseline configuration.

---

## 2. Model

The object detector used in this project is **EfficientDet-D0**.

EfficientDet-D0 is a lightweight one-stage detector that combines:

- a backbone network for feature extraction,
- a bidirectional feature pyramid network for multi-scale feature fusion,
- classification and bounding-box regression heads.

The model is created using the PyTorch `effdet` library with the preset:

```python
tf_efficientdet_d0
```

The model is adapted to SH17 by setting the number of output classes to **17**.

```python
model = create_model(
    "tf_efficientdet_d0",
    bench_task="train",
    bench_labeler=True,
    num_classes=17,
    pretrained=True,
    image_size=(512, 512),
)
```

Pretrained weights are used because training an object detector from scratch requires much more data and computation. Fine-tuning from pretrained weights allows the model to reuse general visual features and adapt them to the SH17 PPE detection task.

---

## 3. Input Size

All images are resized to:

```text
512 x 512
```

This size is selected as a trade-off between detection quality and computational cost.

A larger input size may preserve more details for small PPE objects, but it also increases GPU memory usage and training time. A smaller input size is faster, but it may make small objects harder to detect.

---

## 4. Dataset Loading

The dataset is loaded using a custom SH17 detection dataset class.

During training:

- images are shuffled,
- augmentations are applied,
- batches are created using a custom collate function.

During validation:

- images are not shuffled,
- deterministic evaluation is preserved,
- only baseline transformation is applied.

A custom collate function is required because object detection samples do not have a fixed number of bounding boxes.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=experiment.batch_size,
    shuffle=True,
    collate_fn=collate_detection_batch,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=experiment.batch_size,
    shuffle=False,
    collate_fn=collate_detection_batch,
)
```

---

## 5. Training Configuration

The model is optimized using **AdamW**.

AdamW is selected because it combines adaptive learning-rate behavior with decoupled weight decay, which helps stabilize fine-tuning.

| Hyperparameter | Value |
|---|---|
| Model | EfficientDet-D0 |
| Model preset | `tf_efficientdet_d0` |
| Number of classes | 17 |
| Input size | 512 x 512 |
| Pretrained weights | Yes |
| Optimizer | AdamW |
| Baseline learning rate | `2e-4` |
| Variant learning rate | `1.5e-4` |
| Weight decay | `1e-4` |
| Baseline batch size | 16 |
| Variant batch size | 8 |
| Evaluation interval | Every 5 epochs |

The maximum training length is set to **50 epochs**.  
Validation is performed every **5 epochs** to balance monitoring quality and runtime cost.

---

## 6. Experiment Variants

Four EfficientDet-D0 variants are evaluated.

| Variant | Batch Size | Learning Rate | Early Stopping | Main Purpose |
|---|---:|---:|---|---|
| Baseline | 16 | `2e-4` | No | Reference model |
| Tuned | 8 | `1.5e-4` | Yes | Training stability |
| Oversampling | 8 | `1.5e-4` | Yes | Minority-class learning |
| Zoom-crop | 8 | `1.5e-4` | Yes | Small-object learning |

### 6.1 Baseline Variant

The baseline variant uses the default EfficientDet-D0 setup with:

- input size 512 x 512,
- batch size 16,
- learning rate `2e-4`,
- baseline augmentation,
- no early stopping.

This variant provides a reference point for evaluating later improvements.

### 6.2 Tuned Variant

The tuned variant reduces:

- batch size from 16 to 8,
- learning rate from `2e-4` to `1.5e-4`.

The purpose is to make fine-tuning more stable. Early stopping is enabled as a safeguard.

### 6.3 Oversampling Minority Variant

The oversampling variant repeats training images that contain minority classes.

This is designed to increase the model's exposure to rare PPE-related categories such as:

- helmet,
- safety vest,
- medical suit,
- earmuffs,
- face guard.

Only the training set is expanded. The validation set remains unchanged so that evaluation still reflects the original data distribution.

### 6.4 Zoom-crop Variant

The zoom-crop variant applies scale-focused augmentation.

This is designed to help the model learn small PPE objects more effectively. By randomly zooming and cropping training images, small objects may appear larger and more visible in some training samples.

This variant is especially useful for classes such as:

- gloves,
- glasses,
- helmets,
- earmuffs,
- face guards,
- tools.

---

## 7. Evaluation Metrics

The main evaluation metrics are:

- **mAP@50**
- **mAP@50:95**
- **Precision**
- **Recall**
- **F1-score**

### Why mAP@50 and mAP@50:95 are important

`mAP@50` evaluates object detection performance at an IoU threshold of 0.50. It shows whether the model can generally detect objects under a relaxed localization requirement.

However, `mAP@50` alone is not strict enough. A model may achieve a good mAP@50 score even if the predicted bounding boxes are only roughly aligned with the ground-truth boxes.

Therefore, `mAP@50:95` is used as the main decision metric. It averages mAP across multiple IoU thresholds from 0.50 to 0.95. This makes it a stricter measure because it evaluates both object detection and localization quality.

For SH17, this is especially important because many PPE objects are small and difficult to localize accurately.

---

## 8. Results Summary

The best validation results of the four variants are summarized below.

| Variant | Best Epoch | mAP@50 | mAP@50:95 | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Baseline | 45 | 44.54% | 25.13% | 57.42% | 68.52% |
| Tuned | 45 | 43.79% | 24.61% | ~53-58% | ~68% |
| Oversampling | 15 | 43.03% | 24.47% | ~55-57% | ~68% |
| Zoom-crop | 35 | 45.23% | 26.50% | 58.13% | 68.88% |

---

## 9. Key Findings

### Baseline

The baseline model trains smoothly and reaches its best validation performance around epoch 45. It achieves high precision but only moderate recall. This means the model is reliable when it predicts objects, but it still misses many objects.

### Tuned

The tuned variant provides stable training behavior, but it does not outperform the baseline. Reducing the batch size and learning rate improves stability, but is not enough to significantly improve detection performance.

### Oversampling

The oversampling variant improves early learning by increasing the exposure of minority-class samples. However, validation performance plateaus early, and the final result does not improve over the baseline. This suggests that oversampling alone is not sufficient to solve class imbalance and may increase the risk of overfitting to repeated samples.

### Zoom-crop

The zoom-crop variant achieves the best overall performance. It obtains the highest mAP@50, highest mAP@50:95, highest recall, and highest F1-score among the four variants.

This indicates that scale-focused augmentation is the most effective strategy in this EfficientDet-D0 setup.

---

## 10. Final Conclusion

The best EfficientDet-D0 configuration in this project is the **zoom-crop variant**.

It achieves:

- **45.23% mAP@50**
- **26.50% mAP@50:95**
- **58.13% recall**
- **68.88% F1-score**

Since `mAP@50:95` is the strictest and most reliable metric for object detection quality, it is used as the primary criterion for model selection.

The final result suggests that the most useful strategy for SH17 is not simply reducing the learning rate or repeating minority-class samples. Instead, improving the model's exposure to object scale variation is more effective.

Therefore, zoom-crop augmentation is selected as the best training strategy because it helps EfficientDet-D0 learn better features for small PPE-related objects.



## 11. Notes

- The baseline is useful as a reference configuration.
- The tuned variant improves stability but does not improve validation performance.
- Oversampling helps early training but does not improve generalization.
- Zoom-crop is the best-performing variant because it directly addresses the small-object nature of SH17.
- The best checkpoint should be selected using validation `mAP@50:95`, not simply the final epoch.
