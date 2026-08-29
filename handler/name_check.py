"""A BUILD-TIME assertion: every name a module loads is bound somewhere in that module.

**This is the Python side of `docs/gate-findings.md` F-2026-08-28-15, and it exists because
F-2026-08-29-4 collected on its absence.** The image already asserts that `handler` and
`decodeprobe` IMPORT. An import proves a module parses and that its module-scope imports resolve;
**it does not enter a single function**, so a name used inside one and imported nowhere is
invisible to it. `routec.py` shipped with five uses of `os` and no `import os`, the module imported
cleanly, CI was green on three jobs, and **the wave's most expensive instrument had never executed
once** — found by spending a 4K GPU dispatch on a `NameError`.

**A NAME IS NOT AN INVOCATION AND AN IMPORT IS NOT AN EXECUTION.** That reasoning already changed
the ffmpeg assertion in this file from grepping a capability to running it. This is the same
reasoning applied to the half it had never been applied to.

**WHAT IT DOES NOT DO, stated so nobody reads it as more than it is.** It does not run any path. It
does not resolve attributes — `os.pathh.join` passes, because `os` is bound. It does not know about
names injected at runtime. **It answers exactly one question: is there a name this module reads and
never binds**, which is the question a `NameError` at frame 783 was the answer to.

**Stdlib only and no new dependency**, so it costs a fraction of a second and cannot fail the build
for a reason of its own.

*It ships in the image because `COPY *.py` takes the directory whole; it is never imported or run
by the worker.*
"""

import ast
import builtins
import glob
import sys

#: **The names Python binds in EVERY module's namespace, and `dir(builtins)` is not that set.**
#: It merely overlaps it — `__name__`, `__doc__`, `__spec__`, `__package__` and `__loader__` are
#: attributes of the `builtins` module and so absorbed by accident, while **`__file__` and
#: `__builtins__` are not.** *So `WEIGHTS = os.path.join(os.path.dirname(__file__), "weights")` —
#: an ordinary line — would have failed a good build, and the check would have looked right doing
#: it because four of the five dunders happened to be there.* Found in review; no module here uses
#: `__file__` today, which is exactly why it would have waited for one that did.
MODULE_DUNDERS = frozenset((
    "__file__", "__builtins__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__debug__", "__path__", "__all__",
))

BUILTINS = set(dir(builtins)) | MODULE_DUNDERS

#: PEP 695 type parameters — `def f[T](x: T)` — carry `T` as a STRING on the node, exactly as
#: `match` captures do, while the annotation loads it as a name. **Guarded by `getattr` because
#: these node types do not exist before 3.12 and this image pins 3.11**, so naming them directly
#: would make the check itself the thing that fails. *Same class as the match captures, one
#: version ahead; costs one line to be right about in advance.*
TYPE_PARAM_NODES = tuple(
    node for node in (getattr(ast, name, None)
                      for name in ("TypeVar", "ParamSpec", "TypeVarTuple"))
    if node is not None)


def unbound_names(source):
    """Names loaded in `source` that nothing in it binds, or `None` for a module whose namespace
    cannot be known statically.

    **Binding is read broadly on purpose** — imports, assignments, `def`/`class`, arguments,
    walrus targets, `except ... as`, comprehension variables, MATCH CAPTURES and
    `global`/`nonlocal` — **because a false positive here fails a good build, and that is the
    expensive direction.** *Two were found by probing legal Python shapes rather than by reading:
    `from x import *` and `case {"k": found}` both bind names that never appear as an `ast.Name`
    store, and either would have failed a build the day someone wrote one.*
    """
    tree = ast.parse(source)
    # **A STAR IMPORT MAKES THIS QUESTION UNANSWERABLE, AND SAYING SO BEATS GUESSING.** The names
    # `from x import *` binds are whatever `x` exports at run time, which no AST pass can know —
    # so every name this module reads could be legitimate. **Reported as unknown rather than
    # skipped silently**, because a check that quietly examines nothing is the failure this file
    # exists to prevent, one level up.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
            return None
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound |= {(alias.asname or alias.name) for alias in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound |= set(node.names)
        # **`match` captures bind without ever being an `ast.Name`.** `case {"k": found}` stores
        # into a `MatchMapping`, `case [*rest]` into a `MatchStar`, `case X() as y` into a
        # `MatchAs` — all three carry the name as a plain string attribute.
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif TYPE_PARAM_NODES and isinstance(node, TYPE_PARAM_NODES):
            bound.add(node.name)
    loaded = {node.id for node in ast.walk(tree)
              if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    return sorted(loaded - bound - BUILTINS)


def main():
    modules = sorted(glob.glob("*.py"))
    if not modules:
        # **An empty sweep is a defect and not a pass.** A check that silently examined nothing
        # is the failure this whole file is about, one level up.
        sys.exit("name check found no modules to read — it is being run from the wrong directory")
    faults, unknowable = [], []
    for path in modules:
        with open(path, encoding="utf-8") as handle:
            missing = unbound_names(handle.read())
        if missing is None:
            unknowable.append(path)
            continue
        if missing:
            faults.append("{} uses {} and binds {} nowhere".format(
                path, ", ".join(missing), "them" if len(missing) > 1 else "it"))
    if faults:
        sys.exit("undefined names: " + "; ".join(faults))
    # **The exemption is PRINTED, never silent.** A module this cannot read is a module the build
    # is not asserting anything about, and the reader of a green build should be told which.
    if unknowable:
        print("name check: {} not checked — star import, namespace not knowable statically"
              .format(", ".join(unknowable)))
    print("name check OK: {} modules checked, {} exempt, every loaded name bound".format(
        len(modules) - len(unknowable), len(unknowable)))


if __name__ == "__main__":
    main()
