"""`python -m patchworks` — the same dispatcher the `patchworks` command reaches.

Two routes to one place (#119). The console script is the one to use, since it
needs no `PYTHONPATH` and no memory of a module path; this exists because a
worktree has no console script installed, and because `python -m` is what
someone reaches for when they are not sure the install took.

**This used to print the dome, and that is now `patchworks dome`.** The printout
is unchanged and reachable by name; what changed is that the default is no
longer one subcommand's output masquerading as the whole tool.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
