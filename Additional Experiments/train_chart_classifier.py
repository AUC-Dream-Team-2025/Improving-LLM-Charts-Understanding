import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
import evaluate

class ChartQAFolderDataset(Dataset):
    def __init__(self, annotations_dir, images_dir, processor, label_encoder):
        """
        annotations_dir: folder with JSON annotation files per image
        images_dir: folder with image files
        processor: HuggingFace AutoImageProcessor
        label_encoder: sklearn LabelEncoder fitted on all chart types
        """
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.processor = processor

        self.annotation_files = sorted(os.listdir(annotations_dir))
        self.labels = []
        self.image_files = []

        for ann_file in self.annotation_files:
            ann_path = os.path.join(annotations_dir, ann_file)
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
                chart_type = ann_data["type"]
                img_name = ann_file.replace(".json", ".png")
                self.labels.append(chart_type)
                self.image_files.append(img_name)

        self.label_indices = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = self.label_indices[idx]
        return {"pixel_values": pixel_values, "labels": label}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main():
    # Paths to your folders
    train_ann_dir = "/home/g2/ChartQA Dataset/train/annotations"
    train_img_dir = "/home/g2/ChartQA Dataset/train/png"
    val_ann_dir = "/home/g2/ChartQA Dataset/val/annotations"
    val_img_dir = "/home/g2/ChartQA Dataset/val/png"
    test_ann_dir = "/home/g2/ChartQA Dataset/test/annotations"
    test_img_dir = "/home/g2/ChartQA Dataset/test/png"

    # Load all chart types from train+val+test annotations for consistent label encoding
    all_chart_types = []
    def extract_chart_types(ann_dir):
        files = sorted(os.listdir(ann_dir))
        types = []
        for f in files:
            with open(os.path.join(ann_dir, f), "r") as jf:
                data = json.load(jf)
                types.append(data["type"])
        return types

    all_chart_types.extend(extract_chart_types(train_ann_dir))
    all_chart_types.extend(extract_chart_types(val_ann_dir))
    all_chart_types.extend(extract_chart_types(test_ann_dir))

    label_encoder = LabelEncoder()
    label_encoder.fit(all_chart_types)
    num_labels = len(label_encoder.classes_)
    print(f"Detected {num_labels} unique chart types:", label_encoder.classes_)

    # Load processor and model
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Build datasets
    train_dataset = ChartQAFolderDataset(train_ann_dir, train_img_dir, processor, label_encoder)
    val_dataset = ChartQAFolderDataset(val_ann_dir, val_img_dir, processor, label_encoder)
    test_dataset = ChartQAFolderDataset(test_ann_dir, test_img_dir, processor, label_encoder)

    # Training params
    training_args = TrainingArguments(
        output_dir="./chartqa-dinov2-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        fp16=True,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting fine-tuning DINOv2 on ChartQA...")
    trainer.train()

    print("💾 Saving fine-tuned model and processor...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    print("🎯 Evaluating model on test set...")
    test_metrics = trainer.evaluate(test_dataset)
    print(f"Test accuracy: {test_metrics['eval_accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()
