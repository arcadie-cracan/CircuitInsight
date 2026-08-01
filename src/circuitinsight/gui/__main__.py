import sys

# the launcher, not the app: it shows the loading banner BEFORE the
# heavy imports (matplotlib, sympy, the session layer) are paid for
from .launch import main

if __name__ == "__main__":
    sys.exit(main())
