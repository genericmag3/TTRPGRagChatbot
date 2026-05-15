"""
Entry point for the LocalAIAgent bundled executable.

Starts a Streamlit server on localhost:8501 and opens the browser
automatically. All relative file paths used by the app (assets/, data/)
resolve against the directory that contains this launcher, which is also
where PyInstaller places the bundled files in --onedir mode.
"""
import os
import sys
import threading
import time
import webbrowser


def _exe_dir() -> str:
    """Return the directory that contains the running executable (or this
    script in dev mode). User data and assets live here."""
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller bundle: exe lives one level above _MEIPASS (_internal/)
        # in modern PyInstaller, or at _MEIPASS itself in older versions.
        exe_path = sys.executable
        return os.path.dirname(exe_path)
    return os.path.abspath(os.path.dirname(__file__))


def _bundle_dir() -> str:
    """Return the directory where PyInstaller extracted bundled files
    (sys._MEIPASS), or the project root in dev mode."""
    return getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))


def _ensure_data_dir(exe_dir: str) -> None:
    """Create the data/ directory next to the exe if it doesn't exist.
    The app stores the vector DB and user preferences there at runtime."""
    data_path = os.path.join(exe_dir, "data")
    os.makedirs(data_path, exist_ok=True)


def _open_browser(port: int, delay: float = 4.0) -> None:
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")


def main() -> None:
    exe_dir = _exe_dir()
    bundle_dir = _bundle_dir()

    # Make the exe directory the working directory so that relative paths
    # used by the app (data/, assets/) resolve correctly.
    os.chdir(exe_dir)

    # Copy assets from the bundle into the exe directory if not already there
    # (needed when _MEIPASS != exe_dir, i.e. PyInstaller >= 6).
    if bundle_dir != exe_dir:
        import shutil
        for folder in ("assets", "pages", "src"):
            src = os.path.join(bundle_dir, folder)
            dst = os.path.join(exe_dir, folder)
            if os.path.isdir(src) and not os.path.isdir(dst):
                shutil.copytree(src, dst)
        app_script_src = os.path.join(bundle_dir, "streamlit_app.py")
        app_script_dst = os.path.join(exe_dir, "streamlit_app.py")
        if os.path.isfile(app_script_src) and not os.path.isfile(app_script_dst):
            shutil.copy2(app_script_src, app_script_dst)

    _ensure_data_dir(exe_dir)

    PORT = 8501
    app_script = os.path.join(exe_dir, "streamlit_app.py")

    threading.Thread(target=_open_browser, args=(PORT,), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run",
        app_script,
        f"--server.port={PORT}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    stcli.main()


if __name__ == "__main__":
    main()
