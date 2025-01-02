import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import multiprocessing as mp
from torchvision import models
from copy import deepcopy
import warnings
import csv
import torchvision.transforms as transforms
import timm
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import classification_report
import json
from m_models import *


warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # for multi-GPU setups

# Disable CuDNN heuristics
cudnn.benchmark = False
cudnn.benchmark = False



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
epochs = 30
l_rate = 0.001
batch_size = 32

results = ""

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# weights initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def get_classification_metrics(true_labels, predicted_labels):
    """
    Compute accuracy, precision, recall, and F1-score for each class.

    Args:
        true_labels (list): True class labels.
        predicted_labels (list): Predicted class labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score per class.
    """
    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Accuracy per class
    class_accuracies = (conf_matrix.diagonal() / conf_matrix.sum(axis=1)).tolist()

    # Precision, Recall, F1-score per class
    precision = precision_score(true_labels, predicted_labels, average=None).tolist()
    recall = recall_score(true_labels, predicted_labels, average=None).tolist()
    f1 = f1_score(true_labels, predicted_labels, average=None).tolist()

    # Prepare dictionary
    metrics = {
        "accuracy": class_accuracies,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    metrics_list = []
    for j in range(len(class_accuracies)):
        metrics_list.extend([class_accuracies[j], precision[j], recall[j], f1[j]])

    # metrics_list = [class_accuracies, precision, recall, f1]

    return metrics, metrics_list

def cal_scores(targets, predictions, check=False):
  if check:
    true_labels = [int(t.item()) for t in targets]  # Extract integer values
    predicted_labels = [int(p.item()) for p in predictions]

  scores = []
  # Generate classification report
  report, metrics_list = get_classification_metrics(true_labels, predicted_labels)

  overall_accuracy = round(accuracy_score(true_labels, predicted_labels), 4)* 100

  # Calculate overall precision, recall, and F1-score (weighted average)
  overall_precision = round(precision_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_recall = round(recall_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  overall_f1 = round(f1_score(true_labels, predicted_labels, average='weighted'), 4)* 100
  scores.extend([overall_accuracy, overall_precision, overall_recall, overall_f1])
  scores.extend(metrics_list)

  return {
      'class_metrics' : report,
      'overall_accuracy': overall_accuracy,
      'overall_precision': overall_precision,
      'overall_recall': overall_recall,
      'overall_f1': overall_f1
  }, scores


# Evaluation function 
def evaluate_model(model, data_loader):
    model.eval()
    model.to(device)
    saving_string = ""
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            # print(data.shape, target.shape)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted)
            targets.extend(target)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    saving_string += f"Accuracy: {accuracy:.2f}% \n"
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    dicrt, scores = cal_scores(predictions=predictions, targets=targets, check=True)
    print(dicrt)
    saving_string += json.dumps(dicrt, indent=4)
    return saving_string, scores

def load_data(address, batch_size=64, train=True):
  # Load Fusar dataset
  if train:
    dataset = ImageFolder(root=address, transform=transform_train)
  else: 
    dataset = ImageFolder(root=address, transform=transform_train)

  # Create a dictionary of class names
  class_names = {i: classname for i, classname in enumerate(dataset.classes)}

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2,  # Experiment with different values as recommended above
                            # pin_memory=False, # if torch.cuda.is_available() else False,
                            persistent_workers=True)
  print("Top classes indices:", class_names)

  return data_loader


def get_loaders(dir, data):
    new_class_to_idx = {}
    if data==0:
        new_class_to_idx = {'Cargo': 0, 'Tanker': 1, 'Dredging': 2, 'Fishing': 3, 'Passenger': 4, 'Tug': 5}
    else:
        new_class_to_idx = {'Cargo': 0, 'Fishing': 1, 'Bluk': 2, 'Dredging': 3, 'Container': 4, 'Tanker': 5, 'GeneralCargo': 6, 'Passenger': 7, 'Tug': 8} #fusar


    # Load datasets
    curriculum_data_dir = dir
    easy_dataset = ImageFolder(os.path.join(curriculum_data_dir, "easy"), transform=transform_train)
    moderate_dataset = ImageFolder(os.path.join(curriculum_data_dir, "moderate"), transform=transform_train)
    hard_dataset = ImageFolder(os.path.join(curriculum_data_dir, "hard"), transform=transform_train)
    validation_dataset = ImageFolder(os.path.join(curriculum_data_dir, "validation"), transform=transform_train)
    test_dataset = ImageFolder(os.path.join(curriculum_data_dir, "test"), transform=transform_train)

    # new_class_to_idx = {'Cargo': 0, 'Tanker': 1, 'Dredging': 2, 'Fishing': 3, 'Passenger': 4, 'Tug': 5} # opensar
    # new_class_to_idx = {'Cargo': 0, 'Fishing': 1, 'Bluk': 2, 'Dredging': 3, 'Container': 4, 'Tanker': 5, 'GeneralCargo': 6, 'Passenger': 7, 'Tug': 8} #fusar


    easy_dataset.class_to_idx = new_class_to_idx
    moderate_dataset.class_to_idx = new_class_to_idx
    hard_dataset.class_to_idx = new_class_to_idx
    validation_dataset.class_to_idx = new_class_to_idx
    test_dataset.class_to_idx = new_class_to_idx

    # Create DataLoaders
    batch_size = 32
    easy_loader = DataLoader(easy_dataset, batch_size=batch_size, shuffle=True)
    moderate_loader = DataLoader(moderate_dataset, batch_size=batch_size, shuffle=True)
    hard_loader = DataLoader(hard_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return [easy_loader, moderate_loader, hard_loader, validation_loader, test_loader]

def get_loaders_b(data):
    loaders = []
    fusar_path = "c_fusar_splited"
    loaders.append(load_data(fusar_path + "/train", batch_size=batch_size))
    loaders.append(load_data(fusar_path + "/val", batch_size=batch_size))
    loaders.append(load_data(fusar_path + "/test", batch_size=batch_size))
    
    return loaders

# Function to compute weights
def compute_weights_with_zeros(ratios):
    ratios = torch.tensor(ratios, dtype=torch.float32)
    # Replace zeros with a very small value to avoid division errors
    safe_ratios = torch.where(ratios == 0, torch.tensor(1e-5), ratios)
    weights = 1.0 / safe_ratios  # Inverse of ratios
    normalized_weights = weights / weights.sum()  # Normalize weights
    return normalized_weights.tolist()


def compute_weights_levels(data):
    l1 = [56.5, 43.5, 0, 0, 0, 0, 0, 0, 0]
    l2 = [36.7965368, 29.87012987, 17.31601732, 16.01731602, 0, 0, 0, 0, 0]
    l3 = [25.18518519, 20.37037037, 8.888888889, 8.518518519, 9.62962963, 8.148148148, 7.962962963, 5.925925926, 5.37037037]


    # Compute weights for each level
    weights_l1 = compute_weights_with_zeros(l1)
    weights_l2 = compute_weights_with_zeros(l2)
    weights_l3 = compute_weights_with_zeros(l3)

    return [weights_l1, weights_l2, weights_l3]

def compute_weights_baseline(ratios):
    # Calculate weights as inverse of ratios
    weights = 1.0 / ratios

    # Normalize weights (optional, ensures the weights sum to 1)
    normalized_weights = weights / weights.sum()

    return normalized_weights
# Function to calculate per-class validation loss
def compute_classwise_validation_loss(model, val_loader, loss_fn, num_classes):
    model.eval()
    class_loss = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss_per_sample = loss_fn(outputs, targets)  # Individual losses
            for i in range(num_classes):
                mask = targets == i  # Select samples for class i
                class_loss[i] += loss_per_sample[mask].sum()
                class_count[i] += mask.sum()

    # Avoid division by zero
    class_loss /= (class_count + 1e-5)
    return class_loss

b_opensar_ratio = torch.tensor([0.7035955951, 0.242138782, 0.01884038742, 0.01844235107, 0.008756799788, 0.008226084649])
b_fusar_ratio = torch.tensor([0.5371192893, 0.2490482234, 0.08661167513, 0.04695431472, 0.02062182741, 0.01776649746, 0.0171319797, 0.01300761421, 0.01173857868])
lf_opensar = torch.nn.CrossEntropyLoss(weight=compute_weights_baseline(b_opensar_ratio)).to(device)
lf_fusar = torch.nn.CrossEntropyLoss(weight=compute_weights_baseline(b_fusar_ratio)).to(device)

def train_model(n_model, loaders, device, optimizer, lf, epochs, weights):
    for level, loader in enumerate(loaders[:3]):  # First 3 loaders for training
        lf.weights=weights[level] # for weighted losses
        lf.to(device)
        print(f"Level: {level}")
        for epoch in range(epochs):
            n_model.train()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = n_model(data)
                loss = lf(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()
                
            _, csv_scores = evaluate_and_save_results(n_model, [loaders[3]], device, "", [])
            print("Validation: ", csv_scores)

# Function to evaluate and save results
def evaluate_and_save_results(n_model, loaders, device, results, csv_scores):
    for idx, loader in enumerate(loaders):  # Last 2 loaders for evaluation
        # dataset_type = "Validation" if idx == 0 else "Test"
        # print(f"Evaluating on {dataset_type}")
        str_results, scores = evaluate_model(n_model, loader)
        results += str_results
        csv_scores.append(scores)
    return results, csv_scores

def train_baseline_weights():
    b_fn = "fu"
    classes = 9
    lf = lf_fusar
    c_datasets_b = {f"{b_fn}SAR": get_loaders_b(0)}

    models_ = {
          "Fine_VIT": vit(classes),
          "N-VIT": n_vit(classes),
          "VGG": VGGModel(classes),
          "Fine_VGG": FineTunedVGG(classes),
          "ResNet": ResNetModel(classes),
          "Fine_Resnet": FineTunedResNet(classes)
        }
    
    csv_res = []
    csv1 = []

    for dataset_name, dataset_loader in c_datasets_b.items():
        print("Training on ", dataset_name)
        results += "Training on " + dataset_name + "\n"
        # mix_path = "mix_5"
        train_loader_m = dataset_loader[0]
        val_loader_m = dataset_loader[1]
        test_loader_m = dataset_loader[2]
        for model_name, model in models_.items():
            print("Training using :" , model_name)
            results += "Training using "+model_name + "\n"
                
            n_model = model
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
            n_model.to(device)
            # Training loop
            for epoch in range(epochs):
                n_model.train()
                # print(epoch)
                for batch_idx, data in enumerate(train_loader_m):
                    data, target = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    output = n_model(data)

                    loss = lf(output, target)
                    loss.backward()
                    optimizer.step()
                    
                print("Validation: ", dataset_name)
                results += f"Validation {model_name} on {dataset_name} \n"
                str_results, _ = evaluate_model(n_model, val_loader_m)
                results += str_results
            print("Testing: ", dataset_name)
            results += f"Testing {model_name} on {dataset_name} \n"
            str_results, csv_scores = evaluate_model(n_model, test_loader_m)
            results += str_results
            csv_res.append(csv_scores)
            csv1.append(csv_scores)
                
            # save_model(n_model, save_dir, model_filename)

        # Open the file in write mode ("w") and write the string to it
        with open(f"scores/baseline_w.txt", "w") as f:
            f.write(results)

        fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

        with open(f'scores/baseline_w.csv', 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)

            write.writerow(fields)
            write.writerows(csv1)

def train_curriculum_weights():
    fn = "fu"
    classes = 9
    c_datasets = {f"{fn}SAR": get_loaders(f"c_{fn}sar_ready_3", 1)}

    for i in range(1):
        lf = nn.CrossEntropyLoss()
        weights_curr_fusar = compute_weights_levels(1)

        weights = weights_curr_fusar

        csv_res = []
        csv1 = []
        models_ = {
                "FVIT": vit(classes),
                "N-VIT": n_vit(classes),
                "VGG": VGGModel(classes),
                "Fine_VGG": FineTunedVGG(classes),
                "ResNet": ResNetModel(classes),
                "Fine_Resnet": FineTunedResNet(classes),
                }

        # Main training and evaluation loop
        for model_name, model in models_.items():
            print(f"Training using: {model_name}")
            results += f"Training using {model_name}\n"

            for loader_name, loaders in c_datasets.items():
                print(f"Training on: {loader_name}")
                results += f"Training on {loader_name}\n"

                n_model = model.to(device)
                n_model.apply(initialize_weights)
                optimizer = optim.Adam(n_model.parameters(), lr=l_rate)

                # Train the model
                train_model(n_model, loaders, device, optimizer, lf, epochs, weights)

                # Evaluate and save results
                results, csv1 = evaluate_and_save_results(n_model, [loaders[4]], device, results, csv1)

        # Write results to a text file
        with open(f"scores/cirriculum_w.txt", "w") as f:
            f.write(results)

        # Write scores to a CSV file
        fields = ["Accuracy", "Precision", "Recall", "F1"]*(classes +1)
        with open(f"scores/cirriculum_w.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(csv1)

def train_baseline_weightsU0():
    b_fn = "fu"
    classes = 9
    lf = nn.CrossEntropyLoss()
    c_datasets_b = {f"{b_fn}SAR": get_loaders_b(1)}

    models_ = {
          "Fine_VIT": vit(classes),
          "N-VIT": n_vit(classes),
          "VGG": VGGModel(classes),
          "Fine_VGG": FineTunedVGG(classes),
          "ResNet": ResNetModel(classes),
          "Fine_Resnet": FineTunedResNet(classes)
        }

    csv_res = []
    csv1 = []

    for dataset_name, dataset_loader in c_datasets_b.items():
        print("Training on ", dataset_name)
        results += "Training on " + dataset_name + "\n"
        
        train_loader_m = dataset_loader[0]
        val_loader_m = dataset_loader[1]
        test_loader_m = dataset_loader[2]
        for model_name, model in models_.items():
            print("Training using :" , model_name)
            results += "Training using "+model_name + "\n"
            
            n_model = model
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
            n_model.to(device)
            # Training loop
            for epoch in range(epochs):
                n_model.train()
                # print(epoch)
                for batch_idx, data in enumerate(train_loader_m):
                    data, target = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    output = n_model(data)

                    loss = lf(output, target)
                    loss.backward()
                    optimizer.step()
            
                if epoch%5 == 0:
                    # Compute per-class validation loss
                    val_loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)  # For per-sample loss
                    classwise_val_loss = compute_classwise_validation_loss(model, val_loader_m, val_loss_fn, classes)

                    # Update weights inversely proportional to validation loss
                    new_weights = 1.0 / (classwise_val_loss + 1e-5)  # Avoid division by zero
                    new_weights /= new_weights.sum()  # Normalize weights
                    lf.weight = new_weights  # Update the weights in the loss function

                # else:
                print("Validation: ", dataset_name)
                results += f"Validation {model_name} on {dataset_name} \n"
                str_results, _ = evaluate_model(n_model, val_loader_m)
                results += str_results
            print("Testing: ", dataset_name)
            results += f"Testing {model_name} on {dataset_name} \n"
            str_results, csv_scores = evaluate_model(n_model, test_loader_m)
            results += str_results
            csv_res.append(csv_scores)
            csv1.append(csv_scores)
        
        # save_model(n_model, save_dir, model_filename)

    # Open the file in write mode ("w") and write the string to it
    with open(f"scores/baseline_wu0.txt", "w") as f:
        f.write(results)

    fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

    with open(f'scores/baseline_wu0.csv', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(csv1)

def train_model_wu0(n_model, loaders, device, optimizer, lf, epochs, classes):
    for level, loader in enumerate(loaders[:3]):  # First 3 loaders for training
        # lf.weights=weights[level] # for weighted losses
        lf.to(device)
        print(f"Level: {level}")
        for epoch in range(epochs):
            n_model.train()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = n_model(data)
                loss = lf(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

            if epoch%5 == 0:
                # Compute per-class validation loss
                val_loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)  # For per-sample loss
                classwise_val_loss = compute_classwise_validation_loss(model, loaders[3], val_loss_fn, classes)

                # Update weights inversely proportional to validation loss
                new_weights = 1.0 / (classwise_val_loss + 1e-5)  # Avoid division by zero
                new_weights /= new_weights.sum()  # Normalize weights
                lf.weight = new_weights  # Update the weights in the loss function
            # else: 
                
            _, csv_scores = evaluate_and_save_results(n_model, [loaders[3]], device, "", [])
            print("Validation: ", csv_scores)

def train_curriculum_weightsU0():
    fn = "fu"
    classes = 9
    c_datasets = {f"{fn}SAR": get_loaders(f"c_{fn}sar_ready_3", 1)}
    
    lf = nn.CrossEntropyLoss()
    
    csv1 = []
    models_ = {
            "FVIT": vit(classes),
            "N-VIT": n_vit(classes),
            "VGG": VGGModel(classes),
            "Fine_VGG": FineTunedVGG(classes),
            "ResNet": ResNetModel(classes),
            "Fine_Resnet": FineTunedResNet(classes),
            }

    # Main training and evaluation loop
    for model_name, model in models_.items():
        print(f"Training using: {model_name}")
        results += f"Training using {model_name}\n"

        for loader_name, loaders in c_datasets.items():
            print(f"Training on: {loader_name}")
            results += f"Training on {loader_name}\n"

            n_model = model.to(device)
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)

            # Train the model
            train_model_wu0(n_model, loaders, device, optimizer, lf, epochs, classes)

            # Evaluate and save results
            results, csv1 = evaluate_and_save_results(n_model, [loaders[4]], device, results, csv1)

    # Write results to a text file
    with open(f"scores/cirriculum_wu0.txt", "w") as f:
        f.write(results)

    # Write scores to a CSV file
    fields = ["Accuracy", "Precision", "Recall", "F1"]*(classes +1)
    with open(f"scores/cirriculum_wu0.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(csv1)

def train_baseline_weightsU():
    b_fn = "fu"
    classes = 9
    lf = lf_fusar
    c_datasets_b = {f"{b_fn}SAR": get_loaders_b(1)}


    models_ = {
            "FVIT": vit(classes),
            "N-VIT": n_vit(classes),
            "VGG": VGGModel(classes),
            "Fine_VGG": FineTunedVGG(classes),
            "ResNet": ResNetModel(classes),
            "Fine_Resnet": FineTunedResNet(classes),
            }
    
    csv_res = []
    csv1 = []

    for dataset_name, dataset_loader in c_datasets_b.items():
        print("Training on ", dataset_name)
        results += "Training on " + dataset_name + "\n"
        
        train_loader_m = dataset_loader[0]
        val_loader_m = dataset_loader[1]
        test_loader_m = dataset_loader[2]
        for model_name, model in models_.items():
            print("Training using :" , model_name)
            results += "Training using "+model_name + "\n"
                
            n_model = model
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
            n_model.to(device)
            # Training loop
            for epoch in range(epochs):
                n_model.train()
                for batch_idx, data in enumerate(train_loader_m):
                    data, target = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    output = n_model(data)

                    loss = lf(output, target)
                    loss.backward()
                    optimizer.step()
                
                if epoch%5 == 0:
                    # Compute per-class validation loss
                    val_loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)  # For per-sample loss
                    classwise_val_loss = compute_classwise_validation_loss(model, val_loader_m, val_loss_fn, classes)

                    # Update weights inversely proportional to validation loss
                    new_weights = 1.0 / (classwise_val_loss + 1e-5)  # Avoid division by zero
                    new_weights /= new_weights.sum()  # Normalize weights
                    lf.weight = new_weights  # Update the weights in the loss function

                # else:
                print("Validation: ", dataset_name)
                results += f"Validation {model_name} on {dataset_name} \n"
                str_results, _ = evaluate_model(n_model, val_loader_m)
                results += str_results
            print("Testing: ", dataset_name)
            results += f"Testing {model_name} on {dataset_name} \n"
            str_results, csv_scores = evaluate_model(n_model, test_loader_m)
            results += str_results
            csv_res.append(csv_scores)
            csv1.append(csv_scores)
            
        # save_model(n_model, save_dir, model_filename)

    # Open the file in write mode ("w") and write the string to it
    with open(f"scores/baseline_wu.txt", "w") as f:
        f.write(results)

    fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

    with open(f'scores/baseline_wu.csv', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(csv1)

def train_model_wu(n_model, loaders, device, optimizer, lf, epochs, weights, classes):
    for level, loader in enumerate(loaders[:3]):  # First 3 loaders for training
        lf.weights=weights[level] # for weighted losses
        lf.to(device)
        print(f"Level: {level}")
        for epoch in range(epochs):
            n_model.train()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = n_model(data)
                loss = lf(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

            if epoch%5 == 0:
                # Compute per-class validation loss
                val_loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)  # For per-sample loss
                classwise_val_loss = compute_classwise_validation_loss(n_model, loaders[3], val_loss_fn, classes)

                # Update weights inversely proportional to validation loss
                new_weights = 1.0 / (classwise_val_loss + 1e-5)  # Avoid division by zero
                new_weights /= new_weights.sum()  # Normalize weights
                lf.weight = new_weights  # Update the weights in the loss function
            # else: 
                
            _, csv_scores = evaluate_and_save_results(n_model, [loaders[3]], device, "", [])
            print("Validation: ", csv_scores)

def train_curriculum_weightsU():
    fn = "fu"
    classes = 9
    b_weights_ratio = b_fusar_ratio
    c_datasets = {f"{fn}SAR": get_loaders(f"c_{fn}sar_ready_3", 1)}

    lf = nn.CrossEntropyLoss()
    weights_curr_fusar = compute_weights_levels(1)

    weights = weights_curr_fusar

    csv_res = []
    csv1 = []
    models_ = {
        "FVIT": vit(classes),
        "N-VIT": n_vit(classes),
        "VGG": VGGModel(classes),
        "Fine_VGG": FineTunedVGG(classes),
        "ResNet": ResNetModel(classes),
        "Fine_Resnet": FineTunedResNet(classes),
        }

    # Main training and evaluation loop
    for model_name, model in models_.items():
        print(f"Training using: {model_name}")
        results += f"Training using {model_name}\n"

        for loader_name, loaders in c_datasets.items():
            print(f"Training on: {loader_name}")
            results += f"Training on {loader_name}\n"

            n_model = model.to(device)
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)

            # Train the model
            train_model_wu(n_model, loaders, device, optimizer, lf, epochs, weights, classes)

            # Evaluate and save results
            results, csv1 = evaluate_and_save_results(n_model, [loaders[4]], device, results, csv1)

    # Write results to a text file
    with open(f"scores/cirriculum_wu.txt", "w") as f:
        f.write(results)

    # Write scores to a CSV file
    fields = ["Accuracy", "Precision", "Recall", "F1"]*(classes +1)
    with open(f"scores/cirriculum_wu.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(csv1)

def baseline_norm():
    lf = nn.CrossEntropyLoss()
    b_fn = "fu"
    classes = 9
    c_datasets_b = {f"{b_fn}SAR": get_loaders_b(0)}

    models_ = {
          "Fine_VIT": vit(classes),
          "N-VIT": n_vit(classes),
          "VGG": VGGModel(classes),
          "Fine_VGG": FineTunedVGG(classes),
          "ResNet": ResNetModel(classes),
          "Fine_Resnet": FineTunedResNet(classes)
        }
    
    csv_res = []
    csv1 = []

    for dataset_name, dataset_loader in c_datasets_b.items():
        print("Training on ", dataset_name)
        results += "Training on " + dataset_name + "\n"
        
        train_loader_m = dataset_loader[0]
        val_loader_m = dataset_loader[1]
        test_loader_m = dataset_loader[2]
        for model_name, model in models_.items():
            print("Training using :" , model_name)
            results += "Training using "+model_name + "\n"
                
            n_model = model
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)
            n_model.to(device)
            # Training loop
            for epoch in range(epochs):
                n_model.train()
                for batch_idx, data in enumerate(train_loader_m):
                    data, target = data[0].to(device), data[1].to(device)

                    optimizer.zero_grad()

                    output = n_model(data)

                    loss = lf(output, target)
                    loss.backward()
                    optimizer.step()
                
                print("Validation: ", dataset_name)
                results += f"Validation {model_name} on {dataset_name} \n"
                str_results, _ = evaluate_model(n_model, val_loader_m)
                results += str_results
            print("Testing: ", dataset_name)
            results += f"Testing {model_name} on {dataset_name} \n"
            str_results, csv_scores = evaluate_model(n_model, test_loader_m)
            results += str_results
            csv_res.append(csv_scores)
            csv1.append(csv_scores)
            
        # save_model(n_model, save_dir, model_filename)

    # Open the file in write mode ("w") and write the string to it
    with open("scores/cirriculum_norm.txt", "w") as f:
        f.write(results)

    fields = ["Accuracy", "Precision", "Recall", "F1"] * (classes +1)

    with open('scores/cirriculum_norm.csv', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(csv1)

def train_model_norm(n_model, loaders, device, optimizer, lf, epochs):
    for level, loader in enumerate(loaders[:3]):  # First 3 loaders for training
        
        lf.to(device)
        print(f"Level: {level}")
        for epoch in range(epochs):
            n_model.train()
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)

                # Forward pass
                optimizer.zero_grad()
                output = n_model(data)
                loss = lf(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

            _, csv_scores = evaluate_and_save_results(n_model, [loaders[3]], device, "", [])
            print("Validation: ", csv_scores)

def curriculum_norm():
    fn = "fu"
    classes = 9
    b_weights_ratio = b_fusar_ratio
    c_datasets = {f"{fn}SAR": get_loaders(f"c_{fn}sar_ready_3", 1)}

    lf = nn.CrossEntropyLoss()

    csv_res = []
    csv1 = []
    models_ = {
        "FVIT": vit(classes),
        "N-VIT": n_vit(classes),
        "VGG": VGGModel(classes),
        "Fine_VGG": FineTunedVGG(classes),
        "ResNet": ResNetModel(classes),
        "Fine_Resnet": FineTunedResNet(classes),
        }

    
    # Main training and evaluation loop
    for model_name, model in models.items():
        print(f"Training using: {model_name}")
        results += f"Training using {model_name}\n"

        for loader_name, loaders in c_datasets.items():
            print(f"Training on: {loader_name}")
            results += f"Training on {loader_name}\n"

            n_model = model.to(device)
            n_model.apply(initialize_weights)
            optimizer = optim.Adam(n_model.parameters(), lr=l_rate)

            # Train the model
            train_model(n_model, loaders, device, optimizer, lf, epochs)

            # Evaluate and save results
            results, csv1 = evaluate_and_save_results(n_model, [loaders[4]], device, results, csv1)

    # Write results to a text file
    with open("scores/cirriculum_results_fusar_norm.txt", "w") as f:
        f.write(results)

    # Write scores to a CSV file
    fields = ["Accuracy", "Precision", "Recall", "F1"]*(classes +1)
    with open('scores/cirriculum_scores_fusar_norm.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(csv1)