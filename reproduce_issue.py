
import torch
import torch.nn as nn

def test_squeeze_issue():
    print("--- Testing Squeeze Issue ---")
    
    # Simulate a batch of size 1
    # Output from model: (Batch=1, Dim=1)
    output = torch.randn(1, 1)
    
    # Target: (Batch=1)
    target = torch.randn(1)
    
    criterion = nn.BCELoss()
    
    print(f"Output shape: {output.shape}")
    print(f"Target shape: {target.shape}")
    
    # 1. Reproduce the error
    print("\n1. Testing .squeeze() (Current Implementation):")
    try:
        # squeeze() with no args flattens (1, 1) -> () scalar
        # This causes the mismatch with target (1)
        # However, for BCELoss, inputs must be 0-1. Let's send through sigmoid first.
        probs = torch.sigmoid(output)
        
        # Original code: loss_pin = criteria_pin(out['pin_risk'].squeeze(), y_pin_batch)
        squeezed = probs.squeeze()
        print(f"   Shape after .squeeze(): {squeezed.shape}")
        
        loss = criterion(squeezed, target)
        print("   Success! (Unexpected)")
    except Exception as e:
        print(f"   Caught Expected Error: {e}")

    # 2. Verify the fix
    print("\n2. Testing .squeeze(-1) (Proposed Fix):")
    try:
        probs = torch.sigmoid(output)
        
        # Fix: Only squeeze the last dimension
        squeezed_fixed = probs.squeeze(-1)
        print(f"   Shape after .squeeze(-1): {squeezed_fixed.shape}")
        
        loss = criterion(squeezed_fixed, target)
        print(f"   Success! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   Caught Unexpected Error: {e}")

if __name__ == "__main__":
    test_squeeze_issue()
