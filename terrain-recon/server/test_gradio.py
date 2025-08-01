import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import matplotlib
import base64
import tempfile
import trimesh
from io import BytesIO
import io
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

# Helper function to convert PIL image to base64 data URI
from PIL import Image
import numpy as np

def generate_mesh_from_images(heightmap_img, texture_img, max_height=100.0):
    """
    Convert heightmap (PIL.Image) and texture map (PIL.Image) into 3D mesh data.

    Args:
        heightmap_img (PIL.Image): Grayscale image for heightmap.
        texture_img (PIL.Image): Texture image (color) to map with UV coords.
        max_height (float): Maximum elevation represented in the mesh.

    Returns:
        dict: {
            'vertices': List of (x, y, z) tuples,
            'uvs': List of (u, v) tuples,
            'faces': List of (v0, v1, v2) tuples (index-based),
            'dimensions': (width, height)
        }
    """
    # Ensure both images are the same size
    if heightmap_img.size != texture_img.size:
        raise ValueError("Heightmap and texture must be the same dimensions.")
    
    width, height = heightmap_img.size

    # Convert heightmap to NumPy array and normalize
    height_data = np.asarray(heightmap_img.convert('L'), dtype=np.float32) / 255.0
    height_data *= max_height

    vertices = []
    uvs = []
    faces = []

    for y in range(height):
        for x in range(width):
            z = height_data[y][x]
            vertices.append((x, z, y))  # World position
            uvs.append((x / (width - 1), y / (height - 1)))  # UV coords

    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            i_right = i + 1
            i_bottom = i + width
            i_diag = i_bottom + 1

            # First triangle
            faces.append((i, i_bottom, i_right))

            # Second triangle
            faces.append((i_right, i_bottom, i_diag))

    return {
        'vertices': vertices,
        'uvs': uvs,
        'faces': faces,
        'dimensions': (width, height)
    }

def mesh_to_obj_string(mesh_data):
    vertices = mesh_data['vertices']
    uvs = mesh_data['uvs']
    faces = mesh_data['faces']
    
    lines = []

    # Write vertices
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # Write UVs (texture coordinates)
    for uv in uvs:
        lines.append(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}")  # flip V for OBJ format

    # Write faces (referencing vertex and UV indices, 1-based)
    for f in faces:
        # OBJ face format: f v1/vt1 v2/vt2 v3/vt3
        v1, v2, v3 = f
        lines.append(f"f {v1+1}/{v1+1} {v2+1}/{v2+1} {v3+1}/{v3+1}")

    # Join into OBJ text
    obj_text = '\n'.join(lines)
    return obj_text

# def mesh_to_obj_file(mesh_data):
#     obj_str = mesh_to_obj_string(mesh_data)
#     tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".obj", mode="w")
#     tmp_file.write(obj_str)
#     tmp_file.close()
#     print(tmp_file.name)
#     # return tmp_file.name  # Return file path as string
def mesh_to_obj_file(mesh_data, texture_img):
    obj_str = mesh_to_obj_string(mesh_data)
    
    # Create a temporary folder to hold all files
    temp_dir = tempfile.mkdtemp()
    
    obj_path = os.path.join(temp_dir, "model.obj")
    mtl_path = os.path.join(temp_dir, "model.mtl")
    texture_path = os.path.join(temp_dir, "texture.png")

    # Save texture image
    texture_img.save(texture_path)

    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write(
            "newmtl material0\n"
            "Ka 1.000 1.000 1.000\n"
            "Kd 1.000 1.000 1.000\n"
            "Ks 0.000 0.000 0.000\n"
            "d 1.0\n"
            "illum 2\n"
            "map_Kd texture.png\n"
        )

    # Write OBJ file with reference to MTL
    with open(obj_path, 'w') as f:
        f.write("mtllib model.mtl\n")
        f.write("usemtl material0\n")
        f.write(obj_str)

    return obj_path  # Only return OBJ path; Gradio Model3D will find .mtl and texture if in same folder

# def render_3d_model(heightmap_img, texture_img):
#     mesh = generate_mesh_from_images(heightmap_img, texture_img)
#     obj_file = mesh_to_obj_file(mesh)
#     return obj_file
def render_3d_model(heightmap_img, texture_img):
    mesh = generate_mesh_from_images(heightmap_img, texture_img)
    obj_file_path = mesh_to_obj_file(mesh, texture_img)
    return obj_file_path  # path to .obj file with full material and texture

# def render_3d_model_glb(heightmap_img, texture_img, max_height=100.0):
#     mesh_data = generate_mesh_from_images(heightmap_img, texture_img, max_height)

#     vertices = np.array(mesh_data['vertices'], dtype=np.float32)
#     faces = np.array(mesh_data['faces'], dtype=np.int64)
#     uvs = np.array(mesh_data['uvs'], dtype=np.float32)

#     # Convert heightmap + uvs into a mesh
#     mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
#     mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

#     # Save the texture to a temporary file
#     temp_folder = tempfile.mkdtemp()
#     texture_path = os.path.join(temp_folder, "diffuse.png")
#     texture_img.save(texture_path)

#     material = trimesh.visual.material.PBRMaterial(
#         baseColorTexture=trimesh.visual.texture.TextureVisuals(image=texture_path)
#     )

#     # Apply material (optional: set mesh.visual with material directly)
#     mesh.visual.material = material

#     # Assemble into a scene
#     scene = trimesh.Scene()
#     scene.add_geometry(mesh)

#     # Export to glb
#     glb_path = os.path.join(temp_folder, "terrain.glb")
#     scene.export(glb_path, file_type='glb')
#     return glb_path
def render_3d_model_glb(heightmap_img, texture_img, max_height=60.0):
    mesh_data = generate_mesh_from_images(heightmap_img, texture_img, max_height)
    texture_img_flipped = texture_img.transpose(Image.FLIP_TOP_BOTTOM)

    texture_img = texture_img_flipped
    
    vertices = np.array(mesh_data['vertices'], dtype=np.float32)
    faces = np.array(mesh_data['faces'], dtype=np.int64)
    uvs = np.array(mesh_data['uvs'], dtype=np.float32)

    # Create Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Assign UV coordinates
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    # Save texture to PNG in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tex_file:
        texture_img.save(tex_file.name)
        tex_filepath = tex_file.name

    # Apply texture using visual.material
    mesh.visual.material.image = texture_img  # PIL Image object

    # Build scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    # Write GLB
    glb_path = os.path.join(tempfile.mkdtemp(), "terrain.glb")
    scene.export(glb_path, file_type='glb')

    return glb_path


# --- Model and Presets Loading ---
script_dir = os.path.dirname(os.path.abspath(__file__))
heightmap_model_path = os.path.join(script_dir, '../models/turbo_heightmap_unet_model.pth')
terrain_model_path = os.path.join(script_dir, '../models/turbo_terrain_unet_model.pth')
presets_folder_path = os.path.join(script_dir, './presets')


# device = torch.device("cpu")
device = torch.device("mps")

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

# Load preset image paths
example_paths = []
if os.path.exists(presets_folder_path):
    for filename in os.listdir(presets_folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            example_paths.append(os.path.join(presets_folder_path, filename))
    print(f"Found {len(example_paths)} preset images in {presets_folder_path}")
else:
    print(f"WARNING: Presets folder not found at {presets_folder_path}. No examples will be loaded.")


# Define the image transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
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

    # Normalize terrain colors for facecolors
    normal_map_facecolors = terrain_np / 255.0

    # Create 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.3])  # [X, Y, Z] ratio; make Z axis 30% the scale of X/Y

    
    # Plot the surface with a stride for performance
    # ax.plot_surface(X, Y, Z, facecolors=normal_map_facecolors, rstride=4, cstride=4, linewidth=0, antialiased=False)
    ax.plot_surface(X, Y, Z, facecolors=normal_map_facecolors, rstride=2, cstride=2, linewidth=0, antialiased=False)

    
    # Set view and labels using slider values
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Elevation)')
    ax.set_title("3D Rendered Terrain")
    
    plt.tight_layout()
    return fig

def gaussian_blur(tensor, kernel_size=5, sigma=1.0):
    # Create 1D Gaussian kernel
    def get_gaussian_kernel1d(k, s):
        x = torch.arange(-k//2 + 1., k//2 + 1.)
        kernel = torch.exp(-x**2 / (2*s**2))
        kernel /= kernel.sum()
        return kernel

    kernel_1d = get_gaussian_kernel1d(kernel_size, sigma).to(tensor.device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)

    # Expand to match conv2d weight shape: [out_channels, in_channels, H, W]
    c = tensor.shape[1]
    weight = kernel_2d.expand(c, 1, kernel_size, kernel_size)

    # Apply padding so spatial dims are preserved
    padding = kernel_size // 2
    blurred = F.conv2d(tensor, weight, padding=padding, groups=c)
    return blurred

def predict(input_image_pil, elevation, azimuth):
    """
    Takes a single input image and view angles, generates heightmap 
    and terrain, and creates a 3D plot.
    """
    if input_image_pil is None:
        # Return blank outputs if no image is provided
        blank_image = Image.new('RGB', (256, 256), 'white')
        blank_plot = plt.figure()
        plt.plot([])
        return blank_image, blank_image, blank_plot
        # threejs_html = generate_threejs_html(heightmap_image, terrain_image)
        # return heightmap_image, terrain_image, plot_3d, threejs_html

        # Ensure it's in RGB format
    input_image_pil = input_image_pil.convert("RGB")

    input_tensor = transform_pipeline(input_image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        heightmap_gen_model.eval()
        terrain_gen_model.eval()
        generated_heightmap_tensor = heightmap_gen_model(input_tensor)
        # apply gaussian blur on hm tensor
        generated_heightmap_tensor = gaussian_blur(generated_heightmap_tensor, kernel_size=5, sigma=1.2)

        generated_terrain_tensor = terrain_gen_model(input_tensor)
        generated_terrain_tensor = gaussian_blur(generated_terrain_tensor, kernel_size=5, sigma=1.1)

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

    # threejs_html = generate_threejs_html(heightmap_image, terrain_image)
    # threejs_html = generate_3d_terrain(heightmap_image, terrain_image)
    # object_3d=render_3d_model(heightmap_image, terrain_image)
    object_3d= render_3d_model_glb(heightmap_image, terrain_image)

    return heightmap_image, terrain_image, plot_3d, object_3d

# Create the Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("# 2D and 3D Terrain Generator")
    gr.Markdown("Upload, draw, or choose a preset segmentation map to generate a 2D heightmap, a 2D terrain image, and a 3D rendered terrain.")
    
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Upload & Presets"):
                    input_img_upload = gr.Image(type="pil", label="Input Segmentation Map")
                    if example_paths:
                        gr.Examples(
                            examples=example_paths,
                            inputs=input_img_upload,
                            label="Preset Segmentation Maps"
                        )
                with gr.Tab("Draw"):
                    terrain_colors = [
                    "#118DD7",  # Water 💧
                    "#E1E39B",  # Grassland 🌾
                    "#7FAD7B",  # Forest 🌲
                    "#B97A57",  # Hills ⛰️
                    "#E6C8B5",  # Desert 🏜️
                    "#969696",  # Mountain 🏔️
                    "#C1BEAF"   # Tundra ❄️
                    ]
                    sketchpad = gr.ImageEditor(type="pil", label="Draw Segmentation Map", height=512, width=512, brush=gr.Brush(colors=terrain_colors))

            elevation_slider = gr.Slider(minimum=0, maximum=90, value=30, step=1, label="Elevation Angle")
            azimuth_slider = gr.Slider(minimum=0, maximum=360, value=45, step=1, label="Azimuth Angle")
            btn = gr.Button("Generate")
            
        with gr.Column():
            output_heightmap = gr.Image(type="pil", label="Generated Heightmap (2D)")
            output_terrain = gr.Image(type="pil", label="Generated Terrain (2D)")
            output_plot = gr.Plot(label="Generated Terrain (3D)")
            output_3d_viewer = gr.Model3D(label="Generated 3D Object (not particularly accurate)")
            # output_viewer = gr.HTML(label="Interactive Three.js Terrain")

    # Wrapper function to decide which input to use
    def wrapper_predict(uploaded_img, drawn_img_dict, elevation, azimuth):
        image_to_use = None
        # Check if the user has drawn something meaningful
        if drawn_img_dict and drawn_img_dict["composite"] is not None:
            image_to_use = drawn_img_dict["composite"]
        # Otherwise, fall back to the uploaded image
        elif uploaded_img is not None:
            image_to_use = uploaded_img
        
        return predict(image_to_use, elevation, azimuth)

    # The 'Generate' button triggers the prediction
    btn.click(
        fn=wrapper_predict,
        inputs=[input_img_upload, sketchpad, elevation_slider, azimuth_slider],
        outputs=[output_heightmap, output_terrain, output_plot, output_3d_viewer]
    )
    
    # Allow sliders to update the plot interactively when released
    elevation_slider.release(
        fn=wrapper_predict,
        inputs=[input_img_upload, sketchpad, elevation_slider, azimuth_slider],
        outputs=[output_heightmap, output_terrain, output_plot]
    )
    azimuth_slider.release(
        fn=wrapper_predict,
        inputs=[input_img_upload, sketchpad, elevation_slider, azimuth_slider],
        outputs=[output_heightmap, output_terrain, output_plot]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)