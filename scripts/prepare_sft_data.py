import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen3_agent_sft.training.build_sft_dataset import main

if __name__ == "__main__":
    main()
