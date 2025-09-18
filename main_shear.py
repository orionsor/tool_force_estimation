import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from network_multi import Net, weights_init_normal, weights_init_kaiming
import cv2
from options import Options
from dataset_shear import MultiDirectoryTactileDataset
import sys
from torchvision.transforms.autoaugment import F
from torchvision.utils import save_image
import torch.nn.functional as F

#
# Define an inverse transform to convert normalized tensors back to images
inverse_transform_rgb = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Undo normalization
    transforms.Lambda(lambda x: x.clamp(0, 1))  # Clamp values to [0, 1] range
])

inverse_transform_gray = transforms.Compose([
    transforms.Normalize(mean=[-1], std=[2]),  # Undo normalization
    transforms.Lambda(lambda x: x.clamp(0, 1))  # Clamp values to [0, 1] range
])


# def normalize_force(force_values, min_val, max_val):
#     return (force_values - min_val) / (max_val - min_val)


# def denormalize_force(normalized_values, min_val, max_val):
#     return normalized_values * (max_val - min_val) + min_val

def normalize_force(force_values, mean, std):
    return (force_values - mean) / std

def denormalize_force(normalized_values, mean, std):
    return normalized_values * std + mean


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        diff = torch.abs(predictions - targets)
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


# Add the calculation of min and max force values from the entire dataset
def calculate_mean_std_forces(dataset):
    all_forces = []
    for _, _, _, _, force_value in dataset:
        all_forces.append(force_value)
    mean_force = torch.mean(torch.tensor(all_forces)).item()
    std_force = torch.std(torch.tensor(all_forces)).item()
    return mean_force, std_force

def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img, img_tensor

def plot_standardized_force_distribution(subset, title, save_path, mean_force, std_force, bins=50):
    norm_force_values = []
    for i in range(len(subset)):
        *_, force = subset[i]
        norm_force = (force.item() - mean_force) / std_force
        norm_force_values.append(norm_force)

    norm_force_values = np.array(norm_force_values)
    mean_val = np.mean(norm_force_values)
    std_val = np.std(norm_force_values)

    plt.figure(figsize=(8, 5))
    plt.hist(norm_force_values, bins=bins, edgecolor='black', alpha=0.75)
    plt.title(f'{title}\nMean: {mean_val:.2f}, Std: {std_val:.2f}')
    plt.xlabel('Standardized Force')
    plt.ylabel('Count')
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(np.min(norm_force_values), color='gray', linestyle='dotted', linewidth=1, label='Min')
    plt.axvline(np.max(norm_force_values), color='gray', linestyle='dotted', linewidth=1, label='Max')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved standardized force distribution to: {save_path}")
    plt.close()

def weighted_l1_loss(pred, target):
    # Weight: more importance to higher forces
    weights = 1.0 + 5.0 * (target - target.mean())**2
    loss = weights * torch.abs(pred - target)
    return loss.mean()

def train_epoch(epoch, dataloader, model, mean_force, std_force,is_normal=True):
    # criterion = nn.MSELoss().to(device)
    model.train()
    running_loss = 0.0
    for iter_, (images1_1, images1_2, images2_1, images2_2, forces) in enumerate(dataloader):
        images1_1 = images1_1.to(device)
        images2_1 = images2_1.to(device)
        images1_2 = images1_2.to(device)
        images2_2 = images2_2.to(device)
        forces = forces.to(device).view(-1, 1)  # reshape to match the output dimension

        if is_normal == True:
            forces = normalize_force(forces, mean_force, std_force)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images1_1, images1_2, images2_1, images2_2)
        # loss = weighted_l1_loss(outputs, forces)
        loss = criterion(outputs, forces)
        # loss = torch.sqrt(loss)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if iter_ % opt.print_interval == 0:
            sys.stdout.write(
                '\r' + f"Training Epoch: {epoch + 1}, Iteration: {iter_}/{len(dataloader)}, Batch Loss: {loss:.6f} ")
            sys.stdout.flush()

    epoch_loss = running_loss / len(dataloader.dataset)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return epoch_loss


def evaluate(epoch, dataloader, model, mean_force, std_force, is_normal=True):
    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images1_1, images1_2, images2_1, images2_2, forces in dataloader:
            images1_1 = images1_1.to(device)
            images2_1 = images2_1.to(device)
            images1_2 = images1_2.to(device)
            images2_2 = images2_2.to(device)
            forces = forces.to(device).view(-1, 1)

            if is_normal == True:
                forces = normalize_force(forces, mean_force, std_force)

            outputs = model(images1_1, images1_2, images2_1, images2_2)
            # loss = weighted_l1_loss(outputs, forces)
            loss = criterion(outputs, forces)
            # loss = torch.sqrt(loss)
            val_loss += loss.item()

    val_loss /= len(dataloader.dataset)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

    return val_loss


def calculate_metrics(predicted, actual):
    mae = torch.mean(torch.abs(predicted - actual))
    rmse = torch.sqrt(torch.mean((predicted - actual) ** 2))
    r_squared = 1 - torch.sum((actual - predicted) ** 2) / torch.sum((actual - torch.mean(actual)) ** 2)
    return mae.item(), rmse.item(), r_squared.item()


def test_model(test_loader, model, image_dir, mean_force, std_force, is_rgb=True, is_normal=True,
               inverse_transform=None):
    model.eval()
    # criterion  = nn.L1Loss()
    random_batch_idx = []
    for i in range(5):
        random_batch_idx.append(random.randint(0, len(test_loader) - 1))
    # random_batch_idx = random.randint(0, len(test_loader) - 1)
    random_pair_idx = None

    test_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_r2 = 0.0
    total_samples = 0
    all_test_real_forces = []
    all_test_predicted_forces = []
    with torch.no_grad():
        for images1_1, images1_2, images2_1, images2_2, forces in test_loader:
            images1_1 = images1_1.to(device)
            images2_1 = images2_1.to(device)
            images1_2 = images1_2.to(device)
            images2_2 = images2_2.to(device)
            forces = forces.to(device).view(-1, 1)

            outputs = model(images1_1, images1_2, images2_1, images2_2)

            if is_normal == True:
                normalized_forces = normalize_force(forces, mean_force, std_force)
                loss = weighted_l1_loss(outputs, normalized_forces)
                # loss = criterion(outputs, normalized_forces)
                de_normalized_outputs = denormalize_force(outputs, mean_force, std_force)
                mae, rmse, r2 = calculate_metrics(de_normalized_outputs, forces)
                # Collect real and predicted forces
                all_test_real_forces.extend(forces.cpu().numpy())
                all_test_predicted_forces.extend(de_normalized_outputs.cpu().numpy())
            else:
                loss = weighted_l1_loss(outputs, forces)
                # loss = criterion(outputs, forces)
                mae, rmse, r2 = calculate_metrics(outputs, forces)
                all_test_real_forces.extend(forces.cpu().numpy())
                all_test_predicted_forces.extend(outputs.cpu().numpy())

            total_mae += mae * len(forces)
            total_rmse += rmse * len(forces)
            total_r2 += r2 * len(forces)
            total_samples += len(forces)
            # loss = torch.sqrt(loss)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    avg_mae = total_mae / total_samples
    avg_rmse = total_rmse / total_samples
    avg_r2 = total_r2 / total_samples
    print(f"Test Loss: {test_loss:.4f}")


    return test_loss, avg_mae, avg_rmse, avg_r2, all_test_real_forces, all_test_predicted_forces


def plot_training_forces(model, train_loader, save_dir):
    """
    Plot real and estimated force values for the training set using the best model,
    both in unsorted and sorted order.

    Args:
        model (nn.Module): Trained model.
        train_loader (DataLoader): DataLoader for the training set.
        save_dir (str): Directory to save the plot.
        is_rgb (bool): Whether the images are RGB or grayscale.
    """
    model.eval()
    all_real_forces = []
    all_predicted_forces = []

    with torch.no_grad():
        for images1_1, images1_2, images2_1, images2_2, forces in train_loader:
            images1_1 = images1_1.to(device)
            images2_1 = images2_1.to(device)
            images1_2 = images1_2.to(device)
            images2_2 = images2_2.to(device)
            forces = forces.to(device).view(-1, 1)

            predicted_forces = model(images1_1, images1_2, images2_1, images2_2)

            if is_normal == True:
                normalized_forces = normalize_force(forces, mean_force, std_force)
                # loss = criterion(outputs, normalized_forces)
                de_normalized_outputs = denormalize_force(predicted_forces, mean_force, std_force)
                # Collect real and predicted forces
                all_real_forces.extend(forces.cpu().numpy())
                all_predicted_forces.extend(de_normalized_outputs.cpu().numpy())
            else:
                # loss = criterion(outputs, forces)
                all_real_forces.extend(forces.cpu().numpy())
                all_predicted_forces.extend(predicted_forces.cpu().numpy())

            # Get model predictions

            # # Collect real and predicted forces
            # all_real_forces.extend(forces.cpu().numpy())
            # all_predicted_forces.extend(predicted_forces.cpu().numpy())

    # Convert to numpy arrays for plotting
    all_real_forces = np.array(all_real_forces)
    all_predicted_forces = np.array(all_predicted_forces)

    # Squeeze to remove unnecessary dimensions
    all_real_forces = np.squeeze(all_real_forces)
    all_predicted_forces = np.squeeze(all_predicted_forces)

    force_data = pd.DataFrame({
        'Real_Force': all_real_forces,
        'Predicted_Force': all_predicted_forces
    })
    force_data_path = os.path.join(save_dir, 'force_data_trainset.csv')
    force_data.to_csv(force_data_path, index=False)

    # Plot real vs. estimated force (Unsorted)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(all_real_forces, label='Real Force')
    plt.plot(all_predicted_forces, label='Estimated Force', color='r')
    plt.xlabel('Training Sample Index')
    plt.ylabel('Force')
    plt.title('Real vs Estimated Force on Training Set (Unsorted)')
    plt.legend()

    # Sort the forces by the real force values
    sorted_indices = np.argsort(all_real_forces)
    sorted_real_forces = all_real_forces[sorted_indices]
    sorted_predicted_forces = all_predicted_forces[sorted_indices]

    # Plot sorted real vs. estimated force
    plt.subplot(1, 2, 2)
    plt.plot(sorted_real_forces, label='Real Force')
    plt.plot(sorted_predicted_forces, label='Estimated Force', color='r')
    plt.xlabel('Sorted Sample Index (by Real Force)')
    plt.ylabel('Force')
    plt.title('Real vs Estimated Force on Training Set (Sorted)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_force_comparison.jpg'))
    # plt.show()

    # # Save data for further analysis if needed
    # np.save(os.path.join(save_dir, 'real_forces_training.npy'), all_real_forces)
    # np.save(os.path.join(save_dir, 'predicted_forces_training.npy'), all_predicted_forces)


def generate_data_mappings(root_dir):
    """
    Automatically generate data mappings for all subdirectories within the root directory.
    Assumes each subdirectory contains 'tactile1', 'tactile2', and 'force' subdirectories.
    """
    data_mappings = []
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for subdir in subdirs:
        tactile1_path = os.path.join(root_dir, subdir, 'tactile1')
        tactile2_path = os.path.join(root_dir, subdir, 'tactile2')
        force_path = os.path.join(root_dir, subdir, 'force')

        if all(os.path.exists(p) for p in [tactile1_path, tactile2_path, force_path]):
            data_mappings.append((tactile1_path, tactile2_path, force_path))

    return data_mappings


def plot_force_distribution(subset, title, save_path, bins=50):
    """
    Plots and saves a histogram of force values in a dataset split.

    Args:
        subset (torch.utils.data.Subset): A subset (train/val/test) of your dataset.
        title (str): Title for the plot.
        save_path (str): Path to save the histogram image.
        bins (int): Number of histogram bins.
    """
    force_values = []

    for i in range(len(subset)):
        *_, force = subset[i]
        force_values.append(force.item())

    force_values = np.array(force_values)
    mean_val = np.mean(force_values)
    min_val = np.min(force_values)
    max_val = np.max(force_values)

    plt.figure(figsize=(8, 5))
    plt.hist(force_values, bins=bins, edgecolor='black', alpha=0.75)
    plt.title(f'{title}\nMean: {mean_val:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}')
    plt.xlabel('Y-axis Force')
    plt.ylabel('Count')
    plt.grid(True)
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(min_val, color='gray', linestyle='dotted', linewidth=1, label='Min')
    plt.axvline(max_val, color='gray', linestyle='dotted', linewidth=1, label='Max')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved force distribution plot to: {save_path}")
    plt.close()


if __name__ == '__main__':

    opt = Options().parse()
    # Define data directories

    mapping_root = './data_shear'
    data_mappings = generate_data_mappings(mapping_root)
    is_rgb = opt.is_rgb
    is_normal = opt.is_normal
    is_inverse = opt.is_inverse

    save_dir = opt.save_root
    model_dir = os.path.join(save_dir, opt.save_model_dir)
    image_dir = os.path.join(save_dir, opt.save_images_dir)
    model_save_path = os.path.join(model_dir, 'best_model.pth')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Define transforms
    transform_r = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if is_inverse == True:
        if is_rgb == True:
            inverse_transform = inverse_transform_rgb
        else:
            inverse_transform = inverse_transform_gray
    else:
        inverse_transform = None

    # Specify the number of frames to use
    num_frames = opt.num_frames  # Customize this number as needed

    dataset = MultiDirectoryTactileDataset(data_mappings, transform_r=transform_r, transform_g=transform_t,
                                           num_frames=num_frames, is_rgb=is_rgb)
    # mean_force, std_force = calculate_mean_std_forces(dataset)

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    random.seed(opt.seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    mean_force, std_force = calculate_mean_std_forces(dataset)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)




    # plot_normalized_force_distribution(train_dataset, "Normalized Training Force Distribution",
    # os.path.join(save_dir, 'train_normalized_force_dist.jpg'), min_force, max_force)

    model = Net(is_rgb=is_rgb, is_normal=is_normal)
    model.apply(weights_init_kaiming)
    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Define loss function and optimizer
    # criterion = weighted_l1_loss()
    criterion = nn.L1Loss()
    # criterion = F.smooth_l1_loss()
    # criterion = HuberLoss()
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # Early stopping parameters
    early_stopping_patience = opt.patience
    best_val_loss = np.inf
    patience_counter = 0

    # Training loop
    num_epochs = opt.epochs
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        train_loss = train_epoch(epoch, train_loader, model, mean_force=mean_force, std_force=std_force, is_normal=is_normal)
        train_losses.append(train_loss)

        val_loss = evaluate(epoch, val_loader, model, mean_force=mean_force, std_force=std_force, is_normal=is_normal)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving new best model at epoch {epoch+1} with loss {val_loss:.4f}")
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    # Load the best model
    model.load_state_dict(torch.load(model_save_path))

    # Evaluate on the test set
    model.eval()
    # plot_training_forces(model, train_loader, save_dir)
    test_loss, test_mae, test_rmse, test_r2, all_test_real_forces, all_test_predicted_forces = test_model(test_loader,
                                                                                                          model,
                                                                                                          image_dir,
                                                                                                          mean_force=mean_force, std_force=std_force,
                                                                                                          is_rgb=is_rgb,
                                                                                                          is_normal=is_normal,
                                                                                                          inverse_transform=inverse_transform)

    test_metrics_path = os.path.join(save_dir, 'test_metrics.txt')

    # Save all metrics to the file
    with open(test_metrics_path, 'w') as file:
        file.write(f"Test Loss: {test_loss}\n")
        file.write(f"Test MAE: {test_mae}\n")
        file.write(f"Test RMSE: {test_rmse}\n")
        file.write(f"Test R^2: {test_r2}\n")

    # Plot the real force and estimated force on the test set
    plt.figure(1, figsize=(10, 5))
    plt.plot(all_test_real_forces, label='Real Force')
    plt.plot(all_test_predicted_forces, label='Estimated Force')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Force')
    plt.title('Real vs Estimated Force on Test Set')
    plt.savefig(os.path.join(save_dir, 'force_compare.jpg'))
    # plt.show()

    all_test_real_forces = np.array(all_test_real_forces)
    all_test_predicted_forces = np.array(all_test_predicted_forces)

    # Squeeze the predicted forces array to remove any extra dimensions
    all_test_predicted_forces = np.squeeze(all_test_predicted_forces)
    all_test_real_forces = np.squeeze(all_test_real_forces)

    # Save the force data to a CSV file
    force_data = pd.DataFrame({
        'Real_Force': all_test_real_forces,
        'Predicted_Force': all_test_predicted_forces
    })
    force_data_path = os.path.join(save_dir, 'force_data.csv')
    force_data.to_csv(force_data_path, index=False)

    # Sort the forces by the real force values
    sorted_indices = np.argsort(all_test_real_forces)
    sorted_real_forces = all_test_real_forces[sorted_indices]
    sorted_predicted_forces = all_test_predicted_forces[sorted_indices]

    # Plot the sorted real force and estimated force
    plt.figure(2, figsize=(10, 5))
    plt.plot(sorted_real_forces, label='Real Force')
    plt.plot(sorted_predicted_forces, label='Estimated Force')
    plt.legend()
    plt.xlabel('Sorted Frame (by Real Force)')
    plt.ylabel('Force')
    plt.title('Real vs Estimated Force on Test Set (Sorted by Real Force)')
    plt.savefig(os.path.join(save_dir, 'force_compare_sorted.jpg'))
    # plt.show()

    n = len(train_losses)
    plt.figure(3)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(1, n + 1), torch.tensor(train_losses).cpu(), label='train_loss', color='k')
    plt.legend()
    plt.title('train loss')
    plt.savefig(os.path.join(save_dir, 'train_loss.jpg'))

    # Highlight the lowest validation loss in the plot
    min_val_loss = min(val_losses)
    min_epoch = val_losses.index(min_val_loss) + 1  # +1 to match epoch number



    # Plot validation loss and highlight the lowest point
    plt.figure(4)
    plt.xlabel('epoch')
    plt.ylabel('validation loss')
    plt.title('Validation Loss with Lowest Point Highlighted')
    plt.plot(range(1, n + 1), torch.tensor(val_losses).cpu(), label='validate loss', color='r')
    plt.scatter(min_epoch, min_val_loss, color='blue', label=f'Lowest Loss (Epoch {min_epoch})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_loss_with_highlight.jpg'))

    # Save train and validation loss to a file
    loss_data_path = os.path.join(save_dir, 'loss_data.txt')
    with open(loss_data_path, 'w') as f:
        f.write("Epoch, Train Loss, Validation Loss\n")
        for epoch in range(n):
            f.write(f"{epoch + 1}, {train_losses[epoch]:.6f}, {val_losses[epoch]:.6f}\n")

    plot_training_forces(model, train_loader, save_dir)

