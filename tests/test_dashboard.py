#!/usr/bin/env python3
"""
Quick test to verify Model Accuracy Dashboard is working
"""
import os
import sys
from pathlib import Path

# Resolve project root: tests/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))            # so `import app` works
sys.path.insert(0, str(BASE_DIR / "src"))    # so `import train_multi_disease` works

MODELS_DIR = BASE_DIR / "models"
EVAL_RESULTS_DIR = BASE_DIR / "evaluation_results"

def check_files():
    """Check if all required files exist"""
    print("🔍 Checking required files...")

    checks = {
        "Flask App": BASE_DIR / "app.py",
        "Accuracy Template": BASE_DIR / "templates" / "accuracy.html",
        "Evaluation Script": BASE_DIR / "src" / "comprehensive_evaluation.py",
        "Legacy Pneumonia Model": MODELS_DIR / "model.pth",
        "Multi-Disease Model": MODELS_DIR / "multi_disease_model.pth",
        "Multi-Disease Metadata": MODELS_DIR / "multi_disease_meta.json",
        "Lung Cancer EHR Model": MODELS_DIR / "lung_cancer_model.pkl",
        "Lung Cancer Metadata": MODELS_DIR / "lung_cancer_model_meta.json",
        "Evaluation Report": EVAL_RESULTS_DIR / "evaluation_report.json",
        "Confusion Matrices Image": EVAL_RESULTS_DIR / "confusion_matrices.png",
        "Metrics Comparison Image": EVAL_RESULTS_DIR / "metrics_comparison.png",
        "ROC Curves Image": EVAL_RESULTS_DIR / "roc_curves.png",
    }
    
    all_good = True
    for name, path in checks.items():
        if path.exists():
            size = path.stat().st_size if path.is_file() else "DIR"
            print(f"  ✅ {name}: {path.name} ({size if size == 'DIR' else f'{size:,} bytes'})")
        else:
            print(f"  ❌ {name}: NOT FOUND")
            if "Image" in name or "Report" in name:
                print(f"     → Run evaluation to generate this file")
            else:
                all_good = False
    
    return all_good

def test_flask_import():
    """Test if Flask app can be imported"""
    print("\n🐍 Testing Flask app import...")
    try:
        import app as app_module
        flask_app = app_module.app
        print("  ✅ Flask app imports successfully")
        print(f"  ✅ Routes available: {len(flask_app.url_map._rules)} routes")
        
        # Check for our custom routes
        routes = [rule.rule for rule in flask_app.url_map.iter_rules()]
        required_routes = [
            '/',
            '/dashboard',
            '/how-it-works',
            '/predict',
            '/predict-lung-cancer',
            '/predict-combined',
            '/model-accuracy',
            '/evaluation-image/<path:filename>',
            '/run-evaluation',
        ]
        
        for route in required_routes:
            if "<" in route:
                prefix = route.split("<", 1)[0]
                found = any(r.startswith(prefix) for r in routes)
            else:
                found = route in routes
            if found:
                print(f"  ✅ Route exists: {route}")
            else:
                print(f"  ❌ Route missing: {route}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error importing Flask app: {e}")
        return False


def test_fusion_logic():
    """Verify the combined score is bounded and uses both modalities."""
    print("\n🔗 Testing patient-level fusion logic...")
    try:
        import app as app_module

        fusion = app_module.fuse_patient_risk(
            {"abnormal_probability": 80},
            {"probability": 20},
        )
        if fusion["score"] != 50.0:
            print(f"  ❌ Expected fused score 50.0, got {fusion['score']}")
            return False
        if fusion["components"]["xray_abnormal"] != 80.0:
            print("  ❌ X-ray component not preserved correctly")
            return False
        if fusion["components"]["ehr_lung_cancer"] != 20.0:
            print("  ❌ EHR component not preserved correctly")
            return False

        clamped = app_module.fuse_patient_risk(
            {"abnormal_probability": 150},
            {"probability": -20},
        )
        if not 0 <= clamped["score"] <= 100:
            print("  ❌ Fused score was not clamped into 0-100 range")
            return False

        print("  ✅ Fusion score combines X-ray and EHR probabilities correctly")
        return True
    except Exception as e:
        print(f"  ❌ Fusion test failed: {e}")
        return False


def check_multi_disease_datasets():
    """Check every configured multi-disease class has usable image samples."""
    print("\n🗂️  Checking multi-disease dataset coverage...")
    try:
        from train_multi_disease import collect_samples, validate_class_counts

        _, _, counts = collect_samples(max_per_class=1500, seed=42)
        validate_class_counts(counts)
        for class_name, count in counts.items():
            print(f"  ✅ {class_name}: {count:,} images")
        return True
    except Exception as e:
        print(f"  ❌ Dataset coverage check failed: {e}")
        return False

def main():
    print("="*70)
    print("MODEL ACCURACY DASHBOARD - VERIFICATION TEST")
    print("="*70)
    
    files_ok = check_files()
    flask_ok = test_flask_import()
    fusion_ok = test_fusion_logic()
    datasets_ok = check_multi_disease_datasets()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if files_ok and flask_ok and fusion_ok and datasets_ok:
        print("✅ All checks passed!")
        print("\n🚀 Ready to start:")
        print("   python app.py")
        print("\n📍 Then visit:")
        print("   http://127.0.0.1:5000          — landing")
        print("   http://127.0.0.1:5000/dashboard — analysis app")
        print("\n💡 Use navbar → Metrics for model charts")
        
        # Check if images exist
        images_exist = all([
            (EVAL_RESULTS_DIR / f"{name}.png").exists()
            for name in ["confusion_matrices", "metrics_comparison", "roc_curves"]
        ])
        
        if images_exist:
            print("\n✨ Evaluation images found! Dashboard will show:")
            print("   📊 Confusion Matrices (Visual Heatmaps)")
            print("   📊 Performance Comparison (Bar Chart)")  
            print("   📊 ROC Curves (Model Discrimination)")
        else:
            print("\n⚠️  Evaluation images not found.")
            print("   Click 'Run Model Evaluation' button in dashboard")
            print("   to generate all visualizations (~1-2 minutes)")
        
        return 0
    else:
        print("❌ Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
