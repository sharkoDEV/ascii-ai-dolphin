import sys

from src.ascii_llm.infer import main


if __name__ == "__main__":
    if "--interactive" not in sys.argv:
        sys.argv.append("--interactive")
    main()

