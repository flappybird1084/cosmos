import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import matplotlib
# Set the matplotlib backend to 'Agg' for non-interactive plotting in a server environment.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the DoubleConv and UNet classes exactly as in your notebook
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling path)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (Upsampling path)
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encode
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decode
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x) # Upsampling conv
            skip_connection = skip_connections[idx // 2]
            # Resize if necessary
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            # Concatenate skip connection
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip) # DoubleConv

        return self.final_conv(x)


# --- Model Loading ---
script_dir = os.path.dirname(os.path.abspath(__file__))
heightmap_model_path = os.path.join(script_dir, '../models/turbo_heightmap_unet_model.pth')
terrain_model_path = os.path.join(script_dir, '../models/turbo_terrain_unet_model.pth')

device = torch.device("cpu")

# Initialize models with the correct architecture
heightmap_gen_model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]).to(device)
terrain_gen_model = UNet(in_channels=3, out_channels=3).to(device)

try:
    print(f"Attempting to load heightmap model from: {heightmap_model_path}")
    heightmap_gen_model.load_state_dict(torch.load(heightmap_model_path, map_location=device))
    print(f"Attempting to load terrain model from: {terrain_model_path}")
    terrain_gen_model.load_state_dict(torch.load(terrain_model_path, map_location=device))
    print("--- Models loaded successfully. ---")
except Exception as e:
    print(f"FATAL: Could not load models. Error: {e}")
    exit()

# Define the image transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)), # Reduced size for faster 3D plotting
    transforms.ToTensor(),
])

def generate_3d_plot(heightmap_np, terrain_np, elev, azim):
    """
    Generates a 3D surface plot from a heightmap and a terrain color map.
    """
    heightmap_gray = heightmap_np.squeeze()

    # Prepare for 3D plotting
    rows, cols = heightmap_gray.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = heightmap_gray.astype(np.float32)

    # Normalize and flip terrain colors for facecolors
    normal_map_facecolors = terrain_np / 255.0  # [H, W, 3]
    Z = np.flipud(Z)
    normal_map_facecolors = np.flipud(normal_map_facecolors)

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, facecolors=normal_map_facecolors, rstride=4, cstride=4, linewidth=0, antialiased=False)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Elevation)')
    ax.set_title("3D Rendered Terrain")

    plt.tight_layout()
    return fig
def predict(input_image_pil, elevation, azimuth):
    """
    Takes a PIL image and view angles, generates heightmap and terrain, and creates a 3D plot.
    """
    if input_image_pil is None:
        # Return blank outputs if no image is provided
        blank_image = Image.new('RGB', (256, 256), 'white')
        blank_plot = plt.figure()
        plt.plot([])
        return blank_image, blank_image, blank_plot

    if not isinstance(input_image_pil, Image.Image):
        input_image_pil = Image.fromarray(input_image_pil)

    input_tensor = transform_pipeline(input_image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        heightmap_gen_model.eval()
        terrain_gen_model.eval()
        generated_heightmap_tensor = heightmap_gen_model(input_tensor)
        generated_terrain_tensor = terrain_gen_model(input_tensor)

    # Post-process for 2D image outputs
    heightmap_np = generated_heightmap_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    terrain_np = generated_terrain_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    heightmap_np_viz = (heightmap_np - heightmap_np.min()) / (heightmap_np.max() - heightmap_np.min())
    terrain_np_viz = (terrain_np - terrain_np.min()) / (terrain_np.max() - terrain_np.min())

    heightmap_image = Image.fromarray((heightmap_np_viz * 255).astype(np.uint8).squeeze(), 'L')
    terrain_image = Image.fromarray((terrain_np_viz * 255).astype(np.uint8))
    
    # Generate the 3D plot using the numpy arrays and slider values
    plot_3d = generate_3d_plot(heightmap_np_viz, (terrain_np_viz * 255).astype(np.uint8), elevation, azimuth)
    
    # Close the figure to free up memory
    plt.close(plot_3d)

    return heightmap_image, terrain_image, plot_3d

# Create the Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("# 2D and 3D Terrain Generator")
    gr.Markdown("Upload a segmentation map to generate a 2D heightmap, a 2D terrain image, and a 3D rendered terrain. Use the sliders to change the 3D view.")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Segmentation Map")
            elevation_slider = gr.Slider(minimum=0, maximum=90, value=30, step=1, label="Elevation Angle")
            azimuth_slider = gr.Slider(minimum=0, maximum=360, value=45, step=1, label="Azimuth Angle")
            btn = gr.Button("Generate")
        with gr.Column():
            output_heightmap = gr.Image(type="pil", label="Generated Heightmap (2D)")
            output_terrain = gr.Image(type="pil", label="Generated Terrain (2D)")
            output_plot = gr.Plot(label="Generated Terrain (3D)")

    btn.click(
        fn=predict,
        inputs=[input_img, elevation_slider, azimuth_slider],
        outputs=[output_heightmap, output_terrain, output_plot]
    )
    
    # Allow sliders to update the plot interactively without needing to press the button again
    elevation_slider.change(predict, [input_img, elevation_slider, azimuth_slider], [output_heightmap, output_terrain, output_plot])
    azimuth_slider.change(predict, [input_img, elevation_slider, azimuth_slider], [output_heightmap, output_terrain, output_plot])


# Launch the app
if __name__ == "__main__":
    iface.launch()
