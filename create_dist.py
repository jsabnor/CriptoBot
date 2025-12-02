import os
import shutil
import sys

def create_dist():
    """Creates the 'dist' directory for VPS deployment."""
    dist_dir = 'dist'
    
    # 1. Clean/Create dist directory
    if os.path.exists(dist_dir):
        print(f"🧹 Cleaning existing {dist_dir}...")
        shutil.rmtree(dist_dir)
    os.makedirs(dist_dir)
    print(f"✅ Created {dist_dir} directory")

    # 2. Files to copy (Source -> Dest)
    files_to_copy = [
        ('bot_neural.py', 'bot_neural.py'),
        ('telegram_bot_handler.py', 'telegram_bot_handler.py'),
        ('telegram_notifier.py', 'telegram_notifier.py'),
        ('data_cache.py', 'data_cache.py'),
        ('config.py', 'config.py'),
        ('requirements.txt', 'requirements.txt'),
        ('.env.example', '.env.example'),
    ]

    # 3. Directories to copy
    dirs_to_copy = [
        ('neural_bot', 'neural_bot'),
    ]

    # 4. Specific Models to copy
    # Only copy the winning models to save space and avoid confusion
    models_to_copy = [
        'GENERAL_4h_v2', # For ETH
        'SOL_GROUP_4h',  # For SOL
        'BTC_4h',        # For BTC
    ]

    print("\n📦 Copying files...")
    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dist_dir, dst))
            print(f"  - {src}")
        else:
            print(f"  ⚠️ Warning: {src} not found")

    print("\n📂 Copying directories...")
    for src, dst in dirs_to_copy:
        src_path = src
        dst_path = os.path.join(dist_dir, dst)
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  - {src}")
        else:
            print(f"  ⚠️ Warning: {src} not found")

    print("\n🧠 Copying selected models...")
    models_dest_dir = os.path.join(dist_dir, 'models')
    os.makedirs(models_dest_dir, exist_ok=True)
    
    for model_name in models_to_copy:
        src_path = os.path.join('models', model_name)
        dst_path = os.path.join(models_dest_dir, model_name)
        if os.path.exists(src_path) and os.path.isdir(src_path):
            # Copy entire directory (includes model.keras, scaler.pkl, metadata.json)
            shutil.copytree(src_path, dst_path)
            print(f"  - {model_name} (completo)")
        else:
            print(f"  ⚠️ Warning: Model {model_name} not found")

    print(f"\n✨ Distribution package created in '{dist_dir}/'")
    print("   Next steps: Deploy to VPS and restart bots.")

if __name__ == "__main__":
    create_dist()
