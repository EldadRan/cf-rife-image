"""Route C's decoder: a `cv2.VideoCapture` and the three properties the writer is sized from.

**This is the module that ended the vendored dependency.** `open_source` lived in `pipeline.py`
and took the vendored CLI as its first argument, because `cv2` arrived through
`inference_cli`'s own import so that there was one `cv2` in the process. **That constraint was
about a process that contained SeedVR2, and this one does not** — so the import is direct and
the argument is gone.

**Returns no frame count, and that is deliberate rather than an omission.** `cv2` will answer
`CAP_PROP_FRAME_COUNT` from a container probe while the read is a decode, and where the probe
under-counts the frames are dropped in silence. Route C derives its count from the container's
own duration and rate (`routec.source_frame_count`) and then asserts the decode against it
(`routec.SURPLUS_TOLERANCE_FRAMES`), so a disagreement between the two is refused rather than
delivered short. **Nothing here should grow a frame count**; the check that matters is the one
that compares two independently derived numbers.

**Stills are not route C's.** `open_source` carried a `keep_alpha` branch into
`_open_still_with_alpha`, reached only from the upscale path, and it stayed behind with
`pipeline.py`. A retime of one frame is not a thing this worker does.
"""
import cv2

from errors import INVALID_SOURCE, WorkerError


def open_source(source_path):
    """Open the source and read the shape the encoder will be sized from.

    Returns `(capture, {"fps", "width", "height"})`. **The caller owns the capture's lifetime**
    and must release it; `routec.retime` does so in a `finally`.

    **A capture this REFUSES on is released here, because the caller never gets one to release.**
    Both checks below raise with the capture already constructed, and a refusal that returned
    nothing while holding a demuxer, an FFmpeg context and a file descriptor left them alive
    until the traceback was dropped — with `handle`'s `shutil.rmtree(workdir)` free to remove the
    file underneath a live handle in the meantime. **Inherited verbatim from
    `pipeline.open_source` (`pipeline.py:372-373`) and older than this module**, so it is not
    damage the extraction did; it is repaired here because this is now the only copy.

    The guard is `except BaseException` rather than `except Exception`: a `KeyboardInterrupt` or
    a `SystemExit` arriving between the constructor and the return leaks exactly the same handle,
    and a cleanup that declines to run on the interesting exceptions is not a cleanup.

    **`fps` here is the MEASURED rate and must not be used to plan.** `CAP_PROP_FPS` is
    `avg_frame_rate` — frames divided by duration — while the container's declared cadence is
    `r_frame_rate`, which `probe` reads and which the plan is built from. The two agree on
    anything simple and part company on anything spliced or variable, and reading one where the
    other was meant has already produced one defect on this path: a count derived at the declared
    rate divided against the measured one, nine lines apart, which planned a 23.976 source
    18/455/2 where the contract's arithmetic says 16/458/1.
    """
    capture = cv2.VideoCapture(source_path)
    try:
        if not capture.isOpened():
            raise WorkerError(INVALID_SOURCE, "the decoder could not open the source")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not width or not height:
            raise WorkerError(INVALID_SOURCE, "the decoder reports no dimensions for the source")
    except BaseException:
        capture.release()
        raise
    # **Outside the `try` on purpose.** A `return` inside it would put the success path one
    # editing mistake away from releasing the capture it is handing to the caller.
    return capture, {"fps": fps, "width": width, "height": height}
