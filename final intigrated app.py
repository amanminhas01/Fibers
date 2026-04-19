import cv2
import torch
import joblib
import numpy as np
import gradio as gr
from PIL import Image
import albumentations as A
from scipy import ndimage

from model import ImprovedSegModel
from config import ALL_CLASSES

from scipy.ndimage import convolve
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.morphology import skeletonize
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PATHS
# =========================
SEG_MODEL_PATH = "/home/aman-minhas/Downloads/Retinal_Vessel_Segmentation_using_PyTorch_Semantic_Segmentation/outputsResnet34/best_valid_epoch_dice.pth"
SVM_MODEL_PATH = "/home/aman-minhas/Downloads/Retinal_Vessel_Segmentation_using_PyTorch_Semantic_Segmentation/src/svm_trabecular_model.pkl"
SCALER_PATH    = "/home/aman-minhas/Downloads/Retinal_Vessel_Segmentation_using_PyTorch_Semantic_Segmentation/src/scaler.pkl"

# =========================
# LOAD MODELS
# =========================
seg_model = ImprovedSegModel(num_classes=len(ALL_CLASSES)).to(device)
ckpt = torch.load(SEG_MODEL_PATH, map_location=device)
seg_model.load_state_dict(ckpt['model_state_dict'])
seg_model.eval()

svm = joblib.load(SVM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# ENHANCEMENT
# =========================
from skimage import exposure

def tissue_attenuation_enhancement(img, sigmas=[5,15,30,50,70,100,150], alpha=2.0, beta=0.9, iterations=2):
    enhanced = img.copy()

    for _ in range(iterations):
        temp = enhanced.astype(np.float32) / 255.0

        tissue_maps = [cv2.GaussianBlur(temp, (0,0), s) for s in sigmas]
        tissue = np.mean(tissue_maps, axis=0)

        detail = temp - tissue

        tissue = cv2.bilateralFilter((tissue*255).astype(np.uint8),9,75,75).astype(np.float32)/255.0

        attenuation = np.exp(-beta * tissue)
        tissue_att = tissue * attenuation
        detail_enh = alpha * detail

        enhanced = np.clip(tissue_att + detail_enh, 0, 1)

        enhanced = exposure.rescale_intensity(enhanced, in_range='image', out_range=(0,1))
        enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced

# =========================
# SEGMENTATION
# =========================
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.457,0.433,0.400],
                std=[0.239,0.235,0.239],
                max_pixel_value=255.0)
])

def run_segmentation(gray_img):
    rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    transformed = transform(image=rgb)

    image = transformed["image"]
    image = np.transpose(image, (2,0,1))
    image = torch.tensor(image).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(image)

    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return (pred * 255).astype(np.uint8)

# =========================
# FEATURE HELPERS
# =========================
def fractal_dimension(Z):
    Z = Z > 0
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z,np.arange(0,Z.shape[0],k),axis=0),
                            np.arange(0,Z.shape[1],k),axis=1)
        return len(np.where(S>0)[0])
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p))
    sizes = 2**np.arange(int(np.log2(n)),1,-1)
    counts = [boxcount(Z,int(size)) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts),1)
    return -coeffs[0]

def mask_graph_features(binary):
    skeleton = skeletonize(binary)
    points = np.argwhere(skeleton>0)
    if len(points)<2: return [0]*7

    D = squareform(pdist(points))
    G = nx.Graph()

    for i in range(len(points)):
        G.add_node(i)

    for i in range(len(points)):
        for j in range(i+1,len(points)):
            if D[i,j] <= np.sqrt(2):
                G.add_edge(i,j)

    degrees = [d for n,d in G.degree()] or [0]

    return [
        np.mean(degrees),
        np.max(degrees),
        G.number_of_nodes(),
        G.number_of_edges(),
        nx.density(G),
        np.mean(list(nx.clustering(G).values())) if G.number_of_nodes()>0 else 0,
        nx.number_connected_components(G)
    ]

# =========================
# NEW FEATURE EXTRACTION
# =========================
def extract_features(img, mask):

    _, binary = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    binary = binary//255

    features = []

    # INTENSITY
    features.extend([
        np.mean(img), np.std(img), np.var(img),
        np.median(img), np.min(img), np.max(img),
        np.percentile(img,10), np.percentile(img,25),
        np.percentile(img,75), np.percentile(img,90)
    ])

    # HIST
    hist = cv2.calcHist([img],[0],None,[16],[0,256]).flatten()
    features.extend(hist/(hist.sum()+1e-6))

    # GLCM
    glcm = graycomatrix(img,[1,2,4],
                        [0,np.pi/4,np.pi/2,3*np.pi/4],
                        symmetric=True,normed=True)

    for p in ['contrast','dissimilarity','homogeneity','energy','correlation']:
        features.extend(graycoprops(glcm,p).flatten())

    # LBP
    lbp = local_binary_pattern(img,8,1,'uniform')
    hist,_ = np.histogram(lbp.ravel(), bins=np.arange(0,11), range=(0,10))
    features.extend(hist/(hist.sum()+1e-6))

    # HOG
    hog_feat = hog(img, orientations=9,
                   pixels_per_cell=(16,16),
                   cells_per_block=(2,2),
                   feature_vector=True)
    features.extend(hog_feat[:60])

    # EDGE
    edges = cv2.Canny(img,50,150)
    features.append(np.sum(edges>0)/edges.size)

    # MASK FEATURES
    white_pixels = np.sum(binary)
    total_pixels = binary.size
    black_pixels = total_pixels - white_pixels

    bone_fraction = white_pixels / (total_pixels + 1e-6)
    features.append(bone_fraction)

    dist = ndimage.distance_transform_edt(binary)
    features.append(2*np.mean(dist))

    inv = np.logical_not(binary)
    dist_bg = ndimage.distance_transform_edt(inv)
    features.append(2*np.mean(dist_bg))

    features.append(bone_fraction/(2*np.mean(dist)+1e-6))

    # FRACTAL
    features.append(fractal_dimension(binary))

    # GRAPH
    features.extend(mask_graph_features(binary))

    return np.array(features)

# =========================
# PIPELINE
# =========================
def pipeline(input_image):

    image = np.array(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    enhanced = tissue_attenuation_enhancement(gray)
    mask = run_segmentation(enhanced)

    feats = extract_features(enhanced, mask)
    feats = scaler.transform([feats])

    prob = svm.predict_proba(feats)[0][1]
    pred = 1 if prob > 0.3 else 0

    label = "Osteoporotic" if pred else "Normal"
    confidence = f"{prob*100:.2f}%"

    return image, enhanced, mask, label, confidence

# =========================
# UI
# =========================
interface = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Original"),
        gr.Image(label="Enhanced"),
        gr.Image(label="Mask"),
        gr.Text(label="Prediction"),
        gr.Text(label="Confidence")
    ],
    title="Osteoporosis Detection System"
)

if __name__ == "__main__":
    interface.launch()