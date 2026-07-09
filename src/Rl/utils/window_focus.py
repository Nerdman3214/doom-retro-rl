import subprocess


def find_doom_window():
    """Find the DOOM Retro window ID using xdotool."""
    try:
        result = subprocess.check_output(
            ["xdotool", "search", "--name", "DOOM Retro"],
            text=True
        )
        return result.strip().split("\n")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("DOOM Retro window not found!")
        return None


def get_doom_window_rect(window_id=None):
    """Return (left, top, width, height) of the DOOM Retro window.

    Returns None if the window cannot be found or measured.
    Uses xdotool getwindowgeometry --shell which outputs:
        X=<int>  Y=<int>  WIDTH=<int>  HEIGHT=<int>
    """
    wid = window_id or find_doom_window()
    if not wid:
        return None
    try:
        out = subprocess.check_output(
            ["xdotool", "getwindowgeometry", "--shell", wid],
            text=True,
        )
        vals = {}
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = int(v.strip())
        return (
            vals["X"],
            vals["Y"],
            vals["WIDTH"],
            vals["HEIGHT"],
        )
    except Exception:
        return None


def focus_window(window_id):
    """Activate a window by its xdotool ID."""
    if window_id:
        subprocess.call(
            ["xdotool", "windowactivate", "--sync", window_id]
        )
