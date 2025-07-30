# terrain-recon/server/app.py
import base64
import streamlit.components.v1 as components
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from io import BytesIO


st.title("Random Noise Generator with Canvas")
st.subheader("Canvas Controls")
# Define color options
color_options = {
    "Water": "#118dd7",
    "Grassland": "#e1e373",
    "Forest": "#7fad7b",
    "Hills": "#b97a57",
    "Mountain": "#969696",
    "Tundra": "#c1beaf",
    "Desert": "#e6c8b5"
}

# Color selection with buttons
selected_color = st.session_state.get("selected_color", "#118dd7")

cols = st.columns(len(color_options))
for i, (name, hex_color) in enumerate(color_options.items()):
    if cols[i].button(name):
        selected_color = hex_color
        st.session_state["selected_color"] = hex_color

# Stroke thickness slider
stroke_width = st.slider("Stroke thickness", 15, 35, 2)

# Canvas
st.subheader("Canvas (default water background)")
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",         # Fill color for shapes
    stroke_width=stroke_width,                   # Line thickness
    stroke_color=selected_color,                 # Selected stroke color
    background_color="#118dd7",                  # Canvas background
    update_streamlit=True,
    height=300,
    width=400,
    drawing_mode="freedraw",
    key="canvas"
)
# Button to generate random noise images
if st.button("Generate"):
  st.subheader("Generated Random Noise Images")

  noise_images = []

  # Generate two random 128x128 grayscale noise images
  for i in range(2):
      noise_array = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
      noise_images.append(noise_array)
      noise_image = Image.fromarray(noise_array, mode='L')
      st.image(noise_image, caption=f"Noise Image {i + 1}", use_container_width=False)

  # Use the first image as a heightmap for a 3D surface plot
  heightmap = noise_images[0]
  X, Y = np.meshgrid(np.arange(heightmap.shape[1]), np.arange(heightmap.shape[0]))
  Z = heightmap

  fig = plt.figure(figsize=(6, 4))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=False)
  ax.set_title("3D Heightmap from Noise Image")
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  # Convert matplotlib figure to PNG for display in Streamlit
  buf = BytesIO()
  plt.tight_layout()
  plt.savefig(buf, format="png")
  buf.seek(0)
  st.image(buf, caption="3D Heightmap", use_container_width=True)
  plt.close(fig)


  def pil_image_to_base64(img):
      buffered = BytesIO()
      img.save(buffered, format="PNG")
      return base64.b64encode(buffered.getvalue()).decode()
  # Convert images to base64 and construct <img> src
  base64_img1 = pil_image_to_base64(Image.fromarray(noise_images[0], mode='L'))
  base64_img2 = pil_image_to_base64(Image.fromarray(noise_images[1], mode='L'))

  components.html(f"""
  <!DOCTYPE html>
  <html>
  <head>
      <style>
          body {{
              margin: 0;
              background: #111;
              overflow: hidden;
          }}
          canvas {{
              display: block;
          }}
      </style>
  </head>
  <body>
      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

      <script>
          const scene = new THREE.Scene();
          const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
          camera.position.set(0, 100, 100);
          
          const renderer = new THREE.WebGLRenderer({{ antialias: true }});
          renderer.setSize(window.innerWidth, window.innerHeight);
          document.body.appendChild(renderer.domElement);
          const controls = new THREE.OrbitControls(camera, renderer.domElement);

          const light = new THREE.DirectionalLight(0xffffff, 1);
          light.position.set(0, 1, 1);
          scene.add(light);
          scene.add(new THREE.AmbientLight(0x666666));

          function createHeightMapPlane(imgSrc, offsetX) {{
              const img = new Image();
              img.src = imgSrc;
              img.onload = () => {{
                  const canvas = document.createElement('canvas');
                  canvas.width = img.width;
                  canvas.height = img.height;
                  const ctx = canvas.getContext('2d');
                  ctx.drawImage(img, 0, 0);

                  const imgData = ctx.getImageData(0, 0, img.width, img.height).data;
                  const geometry = new THREE.PlaneGeometry(80, 80, img.width - 1, img.height - 1);

                  for (let i = 0, j = 0; i < imgData.length; i += 4, j++) {{
                      const brightness = imgData[i]; // grayscale
                      geometry.attributes.position.array[j * 3 + 2] = brightness * 0.2; // Z height
                  }}
                  geometry.computeVertexNormals();

                  const material = new THREE.MeshStandardMaterial({{
                      color: 0x2288cc,
                      wireframe: false,
                      side: THREE.DoubleSide,
                      flatShading: true
                  }});
                  const mesh = new THREE.Mesh(geometry, material);
                  mesh.rotation.x = -Math.PI / 2;
                  mesh.position.x = offsetX;
                  scene.add(mesh);
              }};
          }}

          createHeightMapPlane("data:image/png;base64,{base64_img1}", -45);
          createHeightMapPlane("data:image/png;base64,{base64_img2}", 45);

          function animate() {{
              requestAnimationFrame(animate);
              controls.update();
              renderer.render(scene, camera);
          }}
          animate();

          window.addEventListener('resize', () => {{
              const w = window.innerWidth;
              const h = window.innerHeight;
              camera.aspect = w / h;
              camera.updateProjectionMatrix();
              renderer.setSize(w, h);
          }});
      </script>
  </body>
  </html>
  """, height=600)
    # components.html('''
    # <style>
    #     *
    #     {
    #         margin: 0;
    #         padding: 0;
    #     }

    #     html,
    #     body
    #     {
    #         overflow: hidden;
    #         min-height: 700px;
    #     }

    #     .webgl
    #     {
    #         position: fixed;
    #         top: 0;
    #         left: 0;
    #         outline: none;
    #     }
    # </style>

    # <script src="//cdnjs.cloudflare.com/ajax/libs/gsap/3.9.1/gsap.min.js"></script>
    # <!--<script src="//cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.js"></script> -->
    # <!-- <script src="http://threejs.org/examples/js/controls/TrackballControls.js"></script> -->


    # <script type="module">
    #     import * as THREE from 'https://cdn.skypack.dev/three@0.128.0/build/three.module.js';
    #     import { OrbitControls } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/controls/OrbitControls.js';
    #     //import { TrackballControls } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/controls/TrackballControls.js';
          
    #     // Base
    #     // ----------
        
    #     // Initialize scene
    #     const scene = new THREE.Scene()
        
    #     // Initialize camera
    #     const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 60)
        
    #     // Reposition camera
    #     camera.position.set(6, 0, 0)
        
    #     // Initialize renderer
    #     const renderer = new THREE.WebGLRenderer({
    #       alpha: true,
    #       antialias: true
    #     })
        
    #     // Set renderer size
    #     renderer.setSize(window.innerWidth, window.innerHeight)
        
    #     // Append renderer to body
    #     document.body.appendChild(renderer.domElement)
        
    #     // Initialize controls
    #     const controls = new OrbitControls(camera, renderer.domElement)
        
    #     // World
    #     // ----------
        
    #     // Load world texture
    #     const worldTexture = new THREE.TextureLoader().load("https://assets.codepen.io/141041/small-world.jpg")
        
    #     // Initialize world geometry
    #     const worldGeometry = new THREE.SphereGeometry(1, 40, 40)
        
    #     // Initialize world material
    #     const worldMaterial = new THREE.MeshLambertMaterial({
    #       map: worldTexture
    #     })
        
    #     // Initialize world
    #     const world = new THREE.Mesh(worldGeometry, worldMaterial)
        
    #     // Add earth to scene
    #     scene.add(world)
        
    #     // Clouds
    #     // ----------
        
    #     // Load clouds texture
    #     const cloudTexture = new THREE.TextureLoader().load("https://assets.codepen.io/141041/small-world-clouds.png")
        
    #     // Initialize clouds geometry
    #     const cloudGeometry = new THREE.SphereGeometry(1.01, 40, 40)
        
    #     // Initialize clouds material
    #     const cloudMaterial = new THREE.MeshBasicMaterial({
    #       map: cloudTexture,
    #       transparent: true
    #     })
        
    #     // Initialize clouds
    #     const clouds = new THREE.Mesh(cloudGeometry, cloudMaterial)
        
    #     // Add clouds to scene
    #     scene.add(clouds)

    #     // add subtle ambient lighting
    #     const ambientLight = new THREE.AmbientLight(0xbbbbbb);
    #     scene.add(ambientLight);

    #     // directional lighting
    #     const directionalLight = new THREE.DirectionalLight(0xffffff);
    #     directionalLight.position.set(1, 1, 1).normalize();
    #     scene.add(directionalLight);
        
    #     // Animation
    #     // ----------      
        
    #     // Prepare animation loop
    #     function animate() {
    #       // Request animation frame
    #       requestAnimationFrame(animate)
          
    #       // Rotate world
    #       world.rotation.y += 0.0005
          
    #       // Rotate clouds
    #       clouds.rotation.y -= 0.001
          
    #       // Render scene
    #       renderer.render(scene, camera)

    #     }
        
    #     // Animate
    #     animate()
        
    #     // Resize
    #     // ----------
        
    #     // Listen for window resizing
    #     window.addEventListener('resize', () => {
    #       // Update camera aspect
    #       camera.aspect = window.innerWidth / window.innerHeight
          
    #       // Update camera projection matrix
    #       camera.updateProjectionMatrix()
          
    #       // Resize renderer
    #       renderer.setSize(window.innerWidth, window.innerHeight)

    #     });
    # </script>

    # <style>
    #   body{
    #     background: radial-gradient(circle at center, white, rgba(113,129,191,0.5) 50%);
    #   }
    # </style>
    # ''',
    # height=600)

