import torch
from li_4yp.models.statnet import PixelStatNet

def run_statnet_check():
    print("🔬 Checking PixelStatNet Dimensions...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Mock Data
    img = torch.randn(4, 3, 224, 224).to(device)
    
    # 2. Init Model
    model = PixelStatNet(num_classes=[12, 11, 11]).to(device)
    print(f"   -> Model Params: {sum(p.numel() for p in model.parameters()):,}") 
    # Expecting significantly fewer params than EfficientNet (~40k vs 5M)
    
    # 3. Forward Pass
    out = model(img)
    print("   -> Forward Pass Successful.")
    print(f"   -> Output Shapes: Base={out['base'].shape}, Prim={out['primary'].shape}")
    
    print("✅ StatNet is ready.")

if __name__ == "__main__":
    run_statnet_check()