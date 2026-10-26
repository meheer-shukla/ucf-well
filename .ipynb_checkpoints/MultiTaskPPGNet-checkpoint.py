import torch
import torch.nn as nn

class MultiTaskPPGNet(nn.Module):
    """
    Upgraded Multi-Task 1D-CNN Architecture (Matches Methodology Section 4.4)
    
    Shared Trunk: 3x (Conv1D -> BatchNorm1D -> ReLU -> MaxPool1D)
    Heads: HR (Regression), HRV (Regression), Stress (Classification), RR (Regression)
    """
    def __init__(self, num_stress_classes=2):
        super(MultiTaskPPGNet, self).__init__()

        # ==========================================
        # 1. SHARED TRUNK (Feature Extractor)
        # ==========================================
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Global Average Pooling to flatten the sequence
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ==========================================
        # 2. TASK-SPECIFIC HEADS
        # ==========================================
        # Helper function to create a standard head
        def create_head(output_dim):
            return nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, output_dim)
            )

        # HR Head: 1 output (Continuous BPM)
        self.hr_head = create_head(1)
        
        # HRV Head: 2 outputs (RMSSD, SDNN)
        self.hrv_head = create_head(2)
        
        # Stress Head: 2 outputs (Non-Stress, Stress logits)
        self.stress_head = create_head(num_stress_classes)
        
        # RR Head: 1 output (Continuous Breaths Per Minute)
        self.rr_head = create_head(1)

    def forward(self, x):
        """
        Input x: [Batch, 1, Sequence_Length] (e.g., [32, 1, 640])
        """
        # Pass through shared trunk
        x = self.block1(x)  # [B, 32, L/2]
        x = self.block2(x)  # [B, 64, L/4]
        x = self.block3(x)  # [B, 128, L/8]
        
        # Global pooling and flatten
        x = self.global_pool(x) # [B, 128, 1]
        x = x.squeeze(-1)       # [B, 128]

        # Pass through individual heads
        hr = self.hr_head(x)
        hrv = self.hrv_head(x)
        stress = self.stress_head(x)
        rr = self.rr_head(x)

        # Return as a dictionary for clean multi-task loss calculation later
        return {
            'hr': hr,
            'hrv': hrv,
            'stress': stress,
            'rr': rr
        }

# ==========================================
# TEST THE ARCHITECTURE
# ==========================================
if __name__ == "__main__":
    print("Testing Upgraded MultiTaskPPGNet Architecture...")
    
    # Initialize model
    model = MultiTaskPPGNet(num_stress_classes=2)
    
    # Create a dummy batch: Batch=4, Channels=1, Length=640
    dummy_input = torch.randn(4, 1, 640)
    
    # Forward pass
    outputs = model(dummy_input)
    
    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nOutput Shapes:")
    for task, tensor in outputs.items():
        print(f"  {task.upper():<6} -> {tensor.shape}")