import asyncio
import gc
import shutil
import torch
from huggingface_hub import scan_cache_dir

async def cleanup_resources(model_id: str = "google/functiongemma-270m-it"):
    """
    Clear space, remove installed dependencies, free up memory, and remove model weights.
    Should be called after model inference is complete.
    """
    print("\n[Cleanup] Starting cleanup process...")
    
    # 1. Memory Cleanup (Torch/GC)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        gc.collect()
        print("✓ Memory (RAM/VRAM) cleaned")
    except Exception as e:
        print(f"Error during memory cleanup: {e}")

    # 2. Hugging Face Cache Cleanup
    try:
        print(f"Scanning for model {model_id} in cache...")
        hf_cache_info = scan_cache_dir()
        model_found = False
        for repo in hf_cache_info.repos:
            if repo.repo_id == model_id:
                print(f"Removing {model_id} from cache at {repo.repo_path}...")
                shutil.rmtree(repo.repo_path)
                model_found = True
                print(f"✓ Removed {model_id} from local cache")
                break
        
        if not model_found:
            print(f"Model {model_id} not found in cache (or already removed).")
            
    except Exception as e:
        print(f"Error during HF cache cleanup: {e}")

    # 3. UV and Tool Cleanup
    try:
        # Clear uv cache to free up space
        process = await asyncio.create_subprocess_exec(
            "uv", "cache", "clean",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        print("✓ UV cache cleared")

        # Uninstall the mcp server tool environment if it exists via uvx
        process = await asyncio.create_subprocess_exec(
            "uv", "tool", "uninstall", "yahoo-finance-mcp-server",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        print("✓ Yahoo Finance MCP server uninstalled")

    except Exception as e:
        print(f"Error during tool cleanup: {e}")
    
    print("[Cleanup] Process complete.")
