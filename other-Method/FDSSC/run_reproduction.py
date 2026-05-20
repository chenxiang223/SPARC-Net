from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from common.benchmark import main  # noqa: E402


if __name__ == "__main__":
    if "--methods" not in sys.argv:
        sys.argv.extend(["--methods", Path(__file__).resolve().parent.name])
    main()
