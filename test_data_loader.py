#!/usr/bin/env python3
"""
Test script to verify the medical data loader works with your dataset
Run this before attempting full training
"""

import sys
import torch
import logging
from pathlib import Path

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_data_paths():
    """Test if data directories exist"""
    print("=== Testing Data Paths ===")

    train_path = Path("../AUTO_SEGMENTATION_TRAIN")
    test_path = Path("../AUTO_SEGMENTATION_TEST")

    if not train_path.exists():
        print(f"❌ Training data path not found: {train_path.absolute()}")
        print("   Please adjust the path in data_loader.py")
        return False
    else:
        print(f"✅ Training data found: {train_path.absolute()}")

    if not test_path.exists():
        print(f"⚠️  Test data path not found: {test_path.absolute()}")
        print("   Test data is optional, training will still work")
    else:
        print(f"✅ Test data found: {test_path.absolute()}")

    return True

def test_medical_loader():
    """Test the medical data loader"""
    print("\n=== Testing Medical Data Loader ===")

    try:
        from data_loader import MedicalDataLoader
        print("✅ Medical data loader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import medical data loader: {e}")
        return False

    # Initialize loader
    try:
        loader = MedicalDataLoader(
            train_root_dir="../AUTO_SEGMENTATION_TRAIN",
            test_root_dir="../AUTO_SEGMENTATION_TEST",
            target_size=(64, 64, 64),  # Small for testing
            intensity_range=(-1000, 1000),
            normalize=True
        )
        print("✅ Medical data loader initialized")
    except Exception as e:
        print(f"❌ Failed to initialize loader: {e}")
        return False

    # Get case info
    info = loader.get_case_info()
    print(f"📊 Dataset Info:")
    print(f"   Training cases: {info['num_train']} - {info['train_cases']}")
    print(f"   Test cases: {info['num_test']} - {info['test_cases']}")

    if info['num_train'] == 0:
        print("❌ No training cases found!")
        print("   Check your data directory structure")
        return False

    # Test loading one case
    case_name = info['train_cases'][0]
    print(f"\n🧪 Testing load of case: {case_name}")

    try:
        image, mask = loader.load_case(
            loader.train_root_dir,
            case_name,
            image_type="img_fin",
            mask_type="gt_fin"
        )

        print(f"✅ Successfully loaded case:")
        print(f"   Image shape: {image.shape}")
        print(f"   Image dtype: {image.dtype}")
        print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Mask dtype: {mask.dtype}")
        print(f"   Mask unique values: {torch.unique(mask).tolist()}")

        return True

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Check if img_fin.nii.gz and gt_fin.nii.gz exist in your data")
        return False
    except Exception as e:
        print(f"❌ Error loading case: {e}")
        return False

def test_iris_integration():
    """Test integration with IRIS model"""
    print("\n=== Testing IRIS Integration ===")

    try:
        from iris_model import IrisModel
        from episodic_trainer import EpisodicDataset
        print("✅ IRIS components imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import IRIS components: {e}")
        return False

    # Load a small amount of data
    try:
        from data_loader import MedicalDataLoader

        loader = MedicalDataLoader(
            train_root_dir="../AUTO_SEGMENTATION_TRAIN",
            test_root_dir="../AUTO_SEGMENTATION_TEST", 
            target_size=(32, 32, 32),  # Very small for testing
        )

        # Load just 2 cases for testing
        images, masks = loader.load_train_data(max_cases=2)

        if len(images) == 0:
            print("❌ No data loaded")
            return False

        print(f"✅ Loaded {len(images)} cases for testing")

        # Test dataset creation
        dataset = EpisodicDataset(images, masks, episode_length=10)
        print(f"✅ EpisodicDataset created with {len(dataset)} episodes")

        # Test model creation
        model = IrisModel(
            in_channels=1,
            base_channels=8,   # Very small for testing
            embed_dim=64,      # Small for testing
            num_query_tokens=3,
            num_classes=1,
            deep_supervision=False  # Disable for simple testing
        )
        print(f"✅ IRIS model created")

        # Test one forward pass
        sample = dataset[0]
        with torch.no_grad():
            output = model(
                sample['query_image'],
                sample['support_image'], 
                sample['support_mask']
            )
        print(f"✅ Forward pass successful! Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("IRIS MEDICAL DATA LOADER TEST")
    print("=" * 60)

    setup_logging()

    # Run tests
    tests = [
        test_data_paths,
        test_medical_loader, 
        test_iris_integration
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("   python train.py --config config.yaml")
        print("\nOr start with demo mode:")
        print("   python train.py --demo-only")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before training")
        print("\nIf data paths are wrong, update them in:")
        print("   - data_loader.py")
        print("   - train.py (load_medical_datasets function)")

    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)