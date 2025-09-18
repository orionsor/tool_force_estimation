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
from network_multi import Net,weights_init_normal,weights_init_kaiming
import cv2
from options import Options
from dataset_normal import MultiDirectoryTactileDataset
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

def normalize_force(force_values, min_val, max_val):
    return (force_values - min_val) / (max_val - min_val)

def denormalize_force(normalized_values, min_val, max_val):
    return normalized_values * (max_val - min_val) + min_val

class HuberLoss(nn.Module):
  def __init__(self, delta=1.0):
      super().__init__()
      self.delta = delta

  def forward(self, predictions, targets):
      diff = torch.abs(predictions - targets)
      loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
      return loss.mean()

# Add the calculation of min and max force values from the entire dataset
def calculate_min_max_forces(dataset):
    all_forces = []
    for _, _, _, _, force_value in dataset:
        all_forces.append(force_value)
    min_force = min(all_forces)
    max_force = max(all_forces)
    return min_force, max_force

def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    return img, img_tensor


def generate_heatmap(model, img1_1, img1_2, img2_1, img2_2, epsilon=1e-10):
    model.eval()
    img1_1 = img1_1.detach().clone().requires_grad_(True)
    img2_1 = img2_1.detach().clone().requires_grad_(True)
    img1_2 = img1_2.detach().clone().requires_grad_(True)
    img2_2 = img2_2.detach().clone().requires_grad_(True)

    # Forward pass
    output = model(img1_1, img1_2, img2_1, img2_2)

    # Backward pass for img1
    model.zero_grad()
    output.backward(retain_graph=True)
    gradients1_1 = img1_1.grad.data.cpu().numpy()[0]

    model.zero_grad()
    output.backward(retain_graph=True)
    gradients1_2 = img1_2.grad.data.cpu().numpy()[0]

    # Backward pass for img2
    model.zero_grad()
    output.backward(retain_graph=True)
    gradients2_1 = img2_1.grad.data.cpu().numpy()[0]

    model.zero_grad()
    output.backward()
    gradients2_2 = img2_2.grad.data.cpu().numpy()[0]

    # Generate heatmaps
    heatmap1_1 = np.mean(gradients1_1, axis=0)
    heatmap1_1 = np.maximum(heatmap1_1, 0)
    heatmap1_1 = cv2.resize(heatmap1_1, (img1_1.shape[2], img1_1.shape[3]))
    heatmap1_1 = heatmap1_1 - np.min(heatmap1_1)
    heatmap1_1 = heatmap1_1 / (np.max(heatmap1_1) + epsilon)

    heatmap1_2 = np.mean(gradients1_2, axis=0)
    heatmap1_2 = np.maximum(heatmap1_2, 0)
    heatmap1_2 = cv2.resize(heatmap1_2, (img1_2.shape[2], img1_2.shape[3]))
    heatmap1_2 = heatmap1_2 - np.min(heatmap1_2)
    heatmap1_2 = heatmap1_2 / (np.max(heatmap1_2) + epsilon)

    heatmap2_1 = np.mean(gradients2_1, axis=0)
    heatmap2_1 = np.maximum(heatmap2_1, 0)
    heatmap2_1 = cv2.resize(heatmap2_1, (img2_1.shape[2], img2_1.shape[3]))
    heatmap2_1 = heatmap2_1 - np.min(heatmap2_1)
    heatmap2_1 = heatmap2_1 / (np.max(heatmap2_1) + epsilon)

    heatmap2_2 = np.mean(gradients2_2, axis=0)
    heatmap2_2 = np.maximum(heatmap2_2, 0)
    heatmap2_2 = cv2.resize(heatmap2_2, (img2_2.shape[2], img2_2.shape[3]))
    heatmap2_2 = heatmap2_2 - np.min(heatmap2_2)
    heatmap2_2 = heatmap2_2 / (np.max(heatmap2_2) + epsilon)

    return heatmap1_1, heatmap1_2, heatmap2_1, heatmap2_2



# Function to save images with heatmaps
def save_images_with_heatmap(original_img, heatmap, original_output_path, overlay_output_path, is_rgb=True,
                             inverse_transform=None):
    """
    Save images and their heatmaps.

    Args:
        original_img: The original image tensor.
        heatmap: Generated heatmap for the image.
        original_output_path: Path to save the original image.
        overlay_output_path: Path to save the overlayed image.
        is_rgb: Boolean to indicate if the image is RGB or Grayscale.
        inverse_transform: Inverse transform function to get the original image.
    """
    # Apply inverse transform if provided
    if inverse_transform:
        original_img = inverse_transform(original_img)


    save_image(original_img.cpu(),original_output_path)

    # Convert to numpy and save the original image
    # img_np = original_img.detach().cpu().numpy().transpose(1, 2, 0)
    img_np = np.array(transforms.ToPILImage()(original_img.cpu()))
    # Overlay and save the heatmap
    overlayed_img = overlay_heatmap_on_image(img_np, heatmap, is_rgb=is_rgb)
    cv2.imwrite(overlay_output_path, overlayed_img)


def overlay_heatmap_on_image(img, heatmap, alpha=0.6, is_rgb=True):
    # Convert image to three channels if it's single-channel
    if img.ndim == 2:  # If image is (H, W), convert it to (H, W, 3)
        img = np.stack([img] * 3, axis=-1)  # Stack image to create 3 channels

    # Convert heatmap to three channels if it's single-channel
    if heatmap.ndim == 2:  # If heatmap is (H, W), convert it to (H, W, 3)
        heatmap = np.stack([heatmap] * 3, axis=-1)  # Stack heatmap to create 3 channels

    # Apply color map to the heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap[:, :, 0]), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255  # Normalize to [0, 1]

    # Normalize the image to range [0, 1]
    img = np.float32(img) / 255

    # Ensure that both image and heatmap have the same shape
    if heatmap.shape != img.shape:
        raise ValueError(f"Shapes do not match: image shape {img.shape}, heatmap shape {heatmap.shape}")

    # Blend the heatmap and the image
    overlayed_img = heatmap * alpha + img * (1 - alpha)
    overlayed_img = overlayed_img / np.max(overlayed_img)  # Normalize to [0, 1]
    overlayed_img = np.uint8(255 * overlayed_img)  # Scale to [0, 255]
    return overlayed_img

    # Normalize the image to range [0, 1]
    img = np.float32(img) / 255

    # Check that both image and heatmap have the same shape
    if heatmap.shape != img.shape:
        raise ValueError(f"Shapes do not match: image shape {img.shape}, heatmap shape {heatmap.shape}")

    # Blend the heatmap and the image
    overlayed_img = heatmap * alpha + img * (1 - alpha)
    overlayed_img = overlayed_img / np.max(overlayed_img)  # Normalize to [0, 1]
    overlayed_img = np.uint8(255 * overlayed_img)  # Scale to [0, 255]
    return overlayed_img




def train_epoch(epoch, dataloader,model, min_force, max_force,is_normal = True):
    # criterion = nn.MSELoss().to(device)
    model.train()
    running_loss = 0.0
    for iter_,(images1_1, images1_2, images2_1, images2_2, forces) in enumerate(dataloader):
        images1_1 = images1_1.to(device)
        images2_1 = images2_1.to(device)
        images1_2 = images1_2.to(device)
        images2_2 = images2_2.to(device)
        forces = forces.to(device).view(-1, 1)  # reshape to match the output dimension

        if is_normal == True:
            forces = normalize_force(forces, min_force, max_force)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images1_1, images1_2, images2_1, images2_2)
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



def evaluate(epoch, dataloader, model, min_force, max_force,is_normal = True):
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
                forces = normalize_force(forces, min_force, max_force)

            outputs = model(images1_1, images1_2, images2_1, images2_2)
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
def test_model(test_loader, model, image_dir, min_force, max_force, is_rgb=True,is_normal = True, inverse_transform = None):

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
                normalized_forces = normalize_force(forces, min_force, max_force)
                loss = criterion(outputs, normalized_forces)
                de_normalized_outputs = denormalize_force(outputs, min_force, max_force)
                mae, rmse, r2 = calculate_metrics(de_normalized_outputs, forces)
                # Collect real and predicted forces
                all_test_real_forces.extend(forces.cpu().numpy())
                all_test_predicted_forces.extend(de_normalized_outputs.cpu().numpy())
            else:
                loss = criterion(outputs, forces)
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

    for batch_idx, (img1_1,img1_2, img2_1,img2_2, labels) in enumerate(test_loader):
        img1_1, img1_2, img2_1, img2_2, labels = img1_1.to(device), img1_1.to(device), img2_1.to(device), img2_2.to(device), labels.to(device).float()

        if batch_idx in random_batch_idx:
            img1_1.requires_grad = True
            img1_2.requires_grad = True
            img2_1.requires_grad = True
            img2_2.requires_grad = True

            outputs = model(img1_1, img1_2, images2_1, images2_2)


            # Select a random pair of images within the batch
            random_pair_idx = random.randint(0, img1_1.size(0) - 1)
            # random_pair_idx = 0

            # Generate heatmaps for the selected image pair
            heatmap1_1, heatmap1_2, heatmap2_1,heatmap2_2  = generate_heatmap(model, img1_1[random_pair_idx].unsqueeze(0),img1_2[random_pair_idx].unsqueeze(0),
                                                  img2_1[random_pair_idx].unsqueeze(0),img2_2[random_pair_idx].unsqueeze(0), )

            # img1_1np = np.array(transforms.ToPILImage()(img1_1[random_pair_idx].cpu()))
            # img1_2np = np.array(transforms.ToPILImage()(img1_2[random_pair_idx].cpu()))
            #
            #
            # img2_1np = np.array(transforms.ToPILImage()(img2_1[random_pair_idx].cpu()))
            # img2_2np = np.array(transforms.ToPILImage()(img2_2[random_pair_idx].cpu()))

            save_images_with_heatmap(
                original_img=img1_1[random_pair_idx],  # Using the original tensor
                heatmap=heatmap1_1,
                original_output_path=os.path.join(image_dir,
                                                  f'batch_{batch_idx}_original_img1_1_{random_pair_idx}.jpg'),
                overlay_output_path=os.path.join(image_dir, f'batch_{batch_idx}_overlay_img1_1_{random_pair_idx}.jpg'),
                is_rgb=is_rgb,
                inverse_transform=inverse_transform
                # Apply appropriate inverse transform
            )

            save_images_with_heatmap(
                original_img=img1_2[random_pair_idx],  # Using the original tensor
                heatmap=heatmap1_2,
                original_output_path=os.path.join(image_dir,
                                                  f'batch_{batch_idx}_original_img1_2_{random_pair_idx}.jpg'),
                overlay_output_path=os.path.join(image_dir, f'batch_{batch_idx}_overlay_img1_2_{random_pair_idx}.jpg'),
                is_rgb=is_rgb,
                inverse_transform=inverse_transform
                # Apply appropriate inverse transform
            )

            save_images_with_heatmap(
                original_img=img2_1[random_pair_idx],  # Using the original tensor
                heatmap=heatmap2_1,
                original_output_path=os.path.join(image_dir,
                                                  f'batch_{batch_idx}_original_img2_1_{random_pair_idx}.jpg'),
                overlay_output_path=os.path.join(image_dir, f'batch_{batch_idx}_overlay_img2_1_{random_pair_idx}.jpg'),
                is_rgb=is_rgb,
                inverse_transform=inverse_transform
                # Apply appropriate inverse transform
            )

            save_images_with_heatmap(
                original_img=img2_2[random_pair_idx],  # Using the original tensor
                heatmap=heatmap2_2,
                original_output_path=os.path.join(image_dir,
                                                  f'batch_{batch_idx}_original_img2_2_{random_pair_idx}.jpg'),
                overlay_output_path=os.path.join(image_dir, f'batch_{batch_idx}_overlay_img2_2_{random_pair_idx}.jpg'),
                is_rgb=is_rgb,
                inverse_transform=inverse_transform
                # Apply appropriate inverse transform
            )




    return test_loss,avg_mae, avg_rmse, avg_r2, all_test_real_forces, all_test_predicted_forces

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
                normalized_forces = normalize_force(forces, min_force, max_force)
                # loss = criterion(outputs, normalized_forces)
                de_normalized_outputs = denormalize_force(predicted_forces, min_force, max_force)
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
    plt.plot(all_predicted_forces, label='Estimated Force',color = 'r')
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
    plt.plot(sorted_predicted_forces, label='Estimated Force',color = 'r')
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
        tactile1_path = os.path.join(root_dir,subdir, 'tactile1')
        tactile2_path = os.path.join(root_dir,subdir, 'tactile2')
        force_path = os.path.join(root_dir,subdir, 'force')
        
        if all(os.path.exists(p) for p in [tactile1_path, tactile2_path, force_path]):
            data_mappings.append((tactile1_path, tactile2_path, force_path))
    
    return data_mappings




if __name__ == '__main__':

    opt = Options().parse()
    # Define data directories

    mapping_root = './data_normal'
    data_mappings = generate_data_mappings(mapping_root)
    is_rgb = opt.is_rgb
    is_normal = opt.is_normal
    is_inverse = opt.is_inverse

    save_dir = opt.save_root
    model_dir = os.path.join(save_dir, opt.save_model_dir)
    image_dir = os.path.join(save_dir,opt.save_images_dir)
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

    if is_inverse ==True:
        if is_rgb ==True:
            inverse_transform = inverse_transform_rgb
        else:
            inverse_transform = inverse_transform_gray
    else:
        inverse_transform = None

    # Specify the number of frames to use
    num_frames = opt.num_frames  # Customize this number as needed

    
    dataset = MultiDirectoryTactileDataset(data_mappings, transform_r=transform_r, transform_g=transform_t, num_frames=num_frames, is_rgb=is_rgb)
    min_force, max_force = calculate_min_max_forces(dataset)

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    random.seed(opt.seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Net(is_rgb=is_rgb, is_normal=is_normal)
    model.apply(weights_init_kaiming)
    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Define loss function and optimizer
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

        train_loss = train_epoch(epoch, train_loader,model,min_force=min_force,max_force=max_force,is_normal=is_normal)
        train_losses.append(train_loss)



        val_loss = evaluate(epoch,val_loader,model,min_force=min_force,max_force=max_force,is_normal=is_normal)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving new best model at epoch {epoch} with loss {val_loss:.4f}")
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
    test_loss,test_mae, test_rmse, test_r2, all_test_real_forces,all_test_predicted_forces = test_model(test_loader,model,image_dir,min_force=min_force,max_force=max_force,is_rgb=is_rgb, is_normal=is_normal,inverse_transform=inverse_transform)

    test_metrics_path = os.path.join(save_dir, 'test_metrics.txt')

    # Save all metrics to the file
    with open(test_metrics_path, 'w') as file:
        file.write(f"Test Loss: {test_loss}\n")
        file.write(f"Test MAE: {test_mae}\n")
        file.write(f"Test RMSE: {test_rmse}\n")
        file.write(f"Test R^2: {test_r2}\n")



    # Plot the real force and estimated force on the test set
    plt.figure(1,figsize=(10, 5))
    plt.plot(all_test_real_forces, label='Real Force')
    plt.plot(all_test_predicted_forces, label='Estimated Force')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Force')
    plt.title('Real vs Estimated Force on Test Set')
    plt.savefig(os.path.join(save_dir,'force_compare.jpg'))
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
    plt.figure(2,figsize=(10, 5))
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
    

    # plt.figure(4)
    # plt.xlabel('epoch')
    # plt.ylabel('validate loss')
    # plt.title('validate loss ')
    # # plt.ylim((0, 0.5))
    # plt.plot(range(1, n + 1), torch.tensor(val_losses).cpu(), label='validate loss', color='r')
    # plt.legend()
    # plt.savefig(os.path.join(save_dir, 'va_loss.jpg'))
    # # plt.show()

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
        f.write(f"{epoch+1}, {train_losses[epoch]:.6f}, {val_losses[epoch]:.6f}\n") 

    plot_training_forces(model, train_loader, save_dir)

