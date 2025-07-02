from ImageDataset import * 
from torch import nn
from main import UNet
import csv

import torch

# +++++ ===== +++++ ===== +++++ ===== +++++ ===== 

test_path = "dataset/test"
eval_path = "dataset/val"
write_file_path = "examples/write_file.csv"
batch_size = 32
num_models = 5
eval_size = 256
# +++++ ===== +++++ ===== +++++ ===== +++++ ===== 


def main():
    #define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define criterion
    criterion = nn.MSELoss()

    # load image datasets
    test_loader = load_dataset(images_path=test_path, batch_size=batch_size)
    eval_loader = load_dataset(images_path=eval_path, batch_size=batch_size)


    ## evaluate for each model
    # model 
    write_file = []
    for j in range(num_models):
        model = UNet(input_channels=3, output_channels=2)
        print(f'loading model: model_epoch_{j+1}')
        try:
            weights = torch.load(f'output/model_epoch_{j+1}.pth', map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"Model: output/model_epoch_{j+1}.pth does not exist.")
            raise
        
        total_loss = 0
        model.load_state_dict(weights)#.state_dict())
        model.to(device)
        model.eval()

        # Empty GPU mem
        torch.cuda.empty_cache()

        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                images = batch['images']
                images = images.to(device)

                masked_images = batch['masked_images']
                device_masked_images = masked_images.to(device)

                # calculate tloss
                output = model(device_masked_images[:, :, :, :])
                loss = criterion(output, images[:, :2, :, :])
                total_loss += loss.item()

                if i % batch_size == 0:
                    print(f'Batch [{i} / {eval_size}], Total loss: {loss.item():.4f}')
                if i > eval_size:
                    break

        print(f'Average loss: model_epoch_{j+1}: {total_loss / eval_size}')
        write_file.append([f'model_epoch_{j+1}', total_loss / eval_size])
    
    # Write data to CSV file
    with open(write_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(write_file)

def graph(read_file_path):
    x_data = []
    y_data = []
    
    try:
        with open(read_file_path, 'r', newline = '') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                if len(row) >= 2:
                    try:
                        x_data.append((row[0]))
                        y_data.append(float(row[1]))
                    except ValueError:
                        print(f"Value doesn't exist, skipping row {row}")
                else:
                    print(f"Skipping row: {row}")

        
    except FileNotFoundError:
        print(f'File not found: {read_file_path}')
    except Exception as e:
        print(f'An error occured: {e}')

    import matplotlib.pyplot as plt

    plt.plot(x_data, y_data, 'r')
    plt.plot(x_data, y_data, 'o')
    

    plt.xlabel('Model epoch')
    plt.ylabel('MSE error')
    plt.title('Decrease in error vs. Epoch')
    plt.ylim(0.5,1)

    plt.savefig("examples/eval_fig.png")

#main()
graph(write_file_path)

