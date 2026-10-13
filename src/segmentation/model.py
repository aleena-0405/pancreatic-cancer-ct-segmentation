import torch
from monai.networks.nets import UNet


def create_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
    )
    return model


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model().to(device)
    model.eval()

    # Test with a dummy 3D CT patch.
    sample = torch.randn(1, 1, 96, 96, 32).to(device)

    with torch.no_grad():
        prediction = model(sample)

    print("Device:", device)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(prediction.shape))
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    assert prediction.shape == sample.shape
    print("PASS: Model forward test completed.")