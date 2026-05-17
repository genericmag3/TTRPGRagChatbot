# Hosting Guide

This project ships as **two builds from one codebase**, selected at runtime
by the `TTRPG_APP_MODE` environment variable.

| Build  | Mode flag            | Auth | Data layout                  | Launcher             |
|--------|----------------------|------|------------------------------|----------------------|
| Local  | `local` (default)    | none | `data/` (single user)        | `launcher.py`        |
| Remote | `remote`             | yes  | `data/users/<user_id>/`      | `launcher_remote.py` |

The local build is unchanged from before — one user, no login, on
`localhost`. Everything below is about the **remote** (multi-user) build.

## Building

```
python -m PyInstaller TTRPGChatbot.spec          # local single-user
python -m PyInstaller TTRPGChatbot_remote.spec   # remote multi-user host
```

## Running the remote host

```
# from source
TTRPG_APP_MODE=remote python launcher_remote.py
# or run the bundled dist/TTRPGChatbot_remote/TTRPGChatbot_remote.exe
```

On first launch the app shows a one-time **setup screen** to create the
administrator account. Sign in as the admin and open the **Admin** page to:

- **Generate invite links** — single-use, expiring tokens. Send a player
  your server URL with `?invite=<token>` appended.
- **Set a shared registration password** (optional) — when enabled,
  anyone with the password can self-register their own private account.
  Leave it disabled for strictly invite-only access.
- **Manage users** — remove accounts (also revokes their sessions).

Each account gets a fully isolated data directory: its own vector DB,
notes, summary, and settings. Users never see each other's campaigns.

### Environment variables

| Variable               | Default                  | Purpose                                            |
|------------------------|--------------------------|----------------------------------------------------|
| `TTRPG_PORT`           | `8501`                   | Listen port                                        |
| `TTRPG_DATA_DIR`       | `./data`                 | Root for all per-user data + the auth database     |
| `TTRPG_AUTH_DB`        | `<data>/auth/auth.db`    | SQLite credential/session store location           |
| `TTRPG_SSL_CERT`       | `<data>/certs/cert.pem`  | TLS certificate (PEM)                              |
| `TTRPG_SSL_KEY`        | `<data>/certs/key.pem`   | TLS private key (PEM)                              |
| `TTRPG_TLS_HOST`       | `localhost`              | Hostname embedded in the self-signed cert          |
| `TTRPG_ALLOW_INSECURE` | unset                    | `1` = serve plain HTTP (reverse-proxy TLS only)    |

## Networking & security

The remote build applies these defaults:

- **HTTPS is mandatory.** If no certificate is supplied, a self-signed
  one is generated automatically so the login form is never served over
  plain HTTP on the LAN. Browsers will show a warning for self-signed
  certs (expected) — use a reverse proxy for a trusted certificate.
- **CSRF protection on, cross-origin requests off** (Streamlit
  `enableXsrfProtection=true`, `enableCORS=false`).
- **Passwords are bcrypt-hashed** (SHA-256 pre-hash so long passphrases
  are not truncated). Plaintext passwords are never stored.
- **Invite and session tokens are random 256-bit values; only their
  SHA-256 digest is persisted.** A database leak exposes no usable token.
- **Login throttling** — accounts lock briefly after repeated failed
  logins; the counter is persisted so it can't be reset by a restart.
- **Sessions are server-side.** The token lives only in Streamlit's
  per-connection state, never in the URL or browser storage, and is
  re-validated on every interaction.

Operational recommendations:

- Put the host behind a firewall and only expose the chosen port.
- Prefer a reverse proxy with a real certificate for anything beyond a
  trusted LAN.
- Ollama runs locally on the host and is shared by all users for
  inference only; it stores no per-user data.

### Reverse proxy with a trusted certificate (recommended for real use)

Run the app on localhost in HTTP mode and let the proxy handle TLS:

```
TTRPG_APP_MODE=remote TTRPG_ALLOW_INSECURE=1 TTRPG_PORT=8501 python launcher_remote.py
```

Example `Caddyfile` (Caddy obtains and renews a real certificate
automatically):

```
your-domain.example.com {
    reverse_proxy 127.0.0.1:8501
}
```

`TTRPG_ALLOW_INSECURE=1` is **only** appropriate when a proxy like this
terminates TLS in front of the app and the app is not reachable directly
from the network. Without a proxy, leave it unset so the app serves
HTTPS itself.
