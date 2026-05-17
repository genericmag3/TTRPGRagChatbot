"""Self-signed TLS certificate generation for LAN hosting.

The remote build refuses to serve the auth surface over plain HTTP.
When the host has not supplied a real certificate, this generates a
self-signed one so traffic on the local network is still encrypted.
Browsers will warn on a self-signed cert — for a trusted certificate,
front the app with a reverse proxy (see HOSTING.md).
"""
import datetime
import ipaddress
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def ensure_self_signed_cert(
    cert_path: str, key_path: str, host: str = "localhost", days: int = 825
) -> tuple[str, str]:
    """Return ``(cert_path, key_path)``, generating them if missing.

    Idempotent: if both files already exist they are left untouched so a
    restart does not invalidate certificates clients have accepted.
    """
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return cert_path, key_path

    for p in (cert_path, key_path):
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, host)]
    )

    san = [x509.DNSName(host)]
    if host != "localhost":
        san.append(x509.DNSName("localhost"))
    for ip in ("127.0.0.1",):
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    # Best-effort: keep the private key from being world-readable.
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass

    return cert_path, key_path
