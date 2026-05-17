"""Entry point for the REMOTE (multi-user) bundled executable.

Differences from the local launcher:

* Forces ``TTRPG_APP_MODE=remote`` so the app requires authentication
  and isolates each user's data.
* Binds to all interfaces so other machines on the network can connect.
* Enables CSRF protection and disables cross-origin requests.
* Serves over HTTPS. If no certificate is supplied it generates a
  self-signed one (LAN-grade) rather than exposing the login form over
  plain HTTP. Set ``TTRPG_ALLOW_INSECURE=1`` ONLY when a reverse proxy
  terminates TLS in front of this process (see HOSTING.md).

Environment variables
---------------------
TTRPG_PORT            listen port (default 8501)
TTRPG_DATA_DIR        root for all per-user data + auth DB (default ./data)
TTRPG_SSL_CERT        PEM certificate path (default <data>/certs/cert.pem)
TTRPG_SSL_KEY         PEM private key path (default <data>/certs/key.pem)
TTRPG_TLS_HOST        hostname to embed in the self-signed cert
TTRPG_ALLOW_INSECURE  "1" to serve plain HTTP (proxy-terminated TLS only)
"""
import os
import sys


def _exe_dir() -> str:
    if hasattr(sys, "_MEIPASS"):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))


def _bundle_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))


def main() -> None:
    exe_dir = _exe_dir()
    bundle_dir = _bundle_dir()
    os.chdir(exe_dir)

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

    os.environ["TTRPG_APP_MODE"] = "remote"

    data_dir = os.environ.get("TTRPG_DATA_DIR", os.path.join(exe_dir, "data"))
    os.environ["TTRPG_DATA_DIR"] = data_dir
    os.makedirs(data_dir, exist_ok=True)

    port = int(os.environ.get("TTRPG_PORT", "8501"))
    app_script = os.path.join(exe_dir, "streamlit_app.py")

    argv = [
        "streamlit", "run", app_script,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]

    allow_insecure = os.environ.get("TTRPG_ALLOW_INSECURE") == "1"
    if allow_insecure:
        print(
            "WARNING: serving over plain HTTP (TTRPG_ALLOW_INSECURE=1). "
            "Only safe behind a reverse proxy that terminates TLS.",
            file=sys.stderr,
        )
    else:
        cert_path = os.environ.get(
            "TTRPG_SSL_CERT", os.path.join(data_dir, "certs", "cert.pem")
        )
        key_path = os.environ.get(
            "TTRPG_SSL_KEY", os.path.join(data_dir, "certs", "key.pem")
        )
        if not (os.path.isfile(cert_path) and os.path.isfile(key_path)):
            from src.auth.certs import ensure_self_signed_cert
            host = os.environ.get("TTRPG_TLS_HOST", "localhost")
            ensure_self_signed_cert(cert_path, key_path, host=host)
            print(
                f"Generated self-signed certificate at {cert_path}. "
                "Browsers will warn; for a trusted cert use a reverse "
                "proxy (see HOSTING.md).",
                file=sys.stderr,
            )
        argv += [
            f"--server.sslCertFile={cert_path}",
            f"--server.sslKeyFile={key_path}",
        ]

    scheme = "http" if allow_insecure else "https"
    print(f"TTRPG remote host starting on {scheme}://0.0.0.0:{port}", file=sys.stderr)

    from streamlit.web import cli as stcli

    sys.argv = argv
    stcli.main()


if __name__ == "__main__":
    main()
