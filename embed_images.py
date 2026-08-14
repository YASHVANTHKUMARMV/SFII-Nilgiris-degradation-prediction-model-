import os
import re
import base64

md_path = r"d:\Forest Degradation analysis\Forest NDVI modified project\paper\SFII_Comprehensive_Research_Report.md"
base_dir = r"d:\Forest Degradation analysis\Forest NDVI modified project\paper"

with open(md_path, 'r', encoding='utf-8') as f:
    content = f.read()

def replacer(match):
    alt_text = match.group(1)
    img_path = match.group(2)
    
    if img_path.startswith("data:"):
        return match.group(0) # already base64
        
    abs_img_path = os.path.normpath(os.path.join(base_dir, img_path))
    if not os.path.exists(abs_img_path):
        print(f"Warning: Image not found {abs_img_path}")
        return match.group(0)
        
    with open(abs_img_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        
    ext = os.path.splitext(abs_img_path)[1][1:].lower()
    if ext == 'jpg':
        ext = 'jpeg'
    
    print(f"Embedded: {abs_img_path}")
    return f"![{alt_text}](data:image/{ext};base64,{encoded_string})"

new_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replacer, content)

# Add Mermaid workflow diagram
mermaid_diagram = """
### SFII Analytical Workflow

```mermaid
graph TD
    A[Satellite Data Acquisition] --> B[Optical: Sentinel-2 & Landsat]
    A --> C[SAR: Sentinel-1 & LiDAR: GEDI]
    A --> D[Topography & Climate]
    
    B --> E[Spectral Recovery Trajectory SRT]
    C --> F[Structural Biomass Proxy SBP]
    D --> F
    
    B --> H[LandTrendr Disturbance]
    H --> I[Disturbance Memory Function DMF]
    
    B --> G[Ecosystem Resilience Score ERS]
    C --> G
    
    E --> J((SFII Mathematical Integration))
    F --> J
    I --> J
    G --> J
    
    J --> K[Deep Learning Pipeline]
    K --> L[LSTM Sequence Embeddings]
    K --> M[XGBoost Probability Scoring]
    
    L --> N[Spatiotemporal Inference Maps]
    M --> N
```
"""

if "## 2. The Structural Forest Integrity Index (SFII) Methodology" in new_content:
    if "### SFII Analytical Workflow" not in new_content:
        new_content = new_content.replace(
            "## 2. The Structural Forest Integrity Index (SFII) Methodology",
            "## 2. The Structural Forest Integrity Index (SFII) Methodology\n" + mermaid_diagram
        )

with open(md_path, 'w', encoding='utf-8') as f:
    f.write(new_content)
print("Base64 embedding complete.")
