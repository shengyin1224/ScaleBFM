"""Surface failures before cleanup and avoid Isaac's STOP/render wait loop."""
import sys
import traceback


def run_with_cleanup(main, app, resources, get_simulation):
    failure = None
    try:
        main()
    except BaseException:
        failure = sys.exc_info()
        print("STUDENT_FATAL_ERROR", file=sys.stderr, flush=True)
        traceback.print_exception(*failure, file=sys.stderr)
        sys.stderr.flush()

    def disable_stop_wait():
        simulation = get_simulation()
        if simulation is not None:
            # This is the guard used by this IsaacLab version's STOP callback.
            # Disable it before closing either the environment or the app,
            # including when environment construction/cleanup failed.
            simulation._disable_app_control_on_stop_handle = True

    def request_failure_exit():
        # Kit may exit natively inside app.close(), before Python can re-raise.
        # Set its return code first so that a failed run cannot look successful.
        kit_app = getattr(app, "app", None)
        if failure is not None and kit_app is not None:
            kit_app.post_quit(1)

    for cleanup in (disable_stop_wait, resources.close, request_failure_exit, app.close):
        try:
            cleanup()
        except BaseException:
            cleanup_failure = sys.exc_info()
            print("STUDENT_CLEANUP_ERROR", file=sys.stderr, flush=True)
            traceback.print_exception(*cleanup_failure, file=sys.stderr)
            sys.stderr.flush()
            if failure is None:
                failure = cleanup_failure
    if failure is not None:
        raise failure[1].with_traceback(failure[2])
