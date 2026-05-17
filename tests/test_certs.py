"""Tests for self-signed TLS certificate generation."""
import os

from cryptography import x509

from src.auth.certs import ensure_self_signed_cert


class TestEnsureSelfSignedCert:
    def test_generates_cert_and_key(self, tmp_path):
        cert = str(tmp_path / "certs" / "cert.pem")
        key = str(tmp_path / "certs" / "key.pem")
        ensure_self_signed_cert(cert, key, host="localhost")
        assert os.path.isfile(cert)
        assert os.path.isfile(key)
        assert b"BEGIN CERTIFICATE" in open(cert, "rb").read()
        assert b"PRIVATE KEY" in open(key, "rb").read()

    def test_certificate_is_parseable_and_has_san(self, tmp_path):
        cert = str(tmp_path / "cert.pem")
        key = str(tmp_path / "key.pem")
        ensure_self_signed_cert(cert, key, host="myhost")
        loaded = x509.load_pem_x509_certificate(open(cert, "rb").read())
        san = loaded.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        dns_names = san.get_values_for_type(x509.DNSName)
        assert "myhost" in dns_names
        assert "localhost" in dns_names

    def test_is_idempotent(self, tmp_path):
        cert = str(tmp_path / "cert.pem")
        key = str(tmp_path / "key.pem")
        ensure_self_signed_cert(cert, key)
        first = open(cert, "rb").read()
        ensure_self_signed_cert(cert, key)
        # Existing files must be preserved so accepted certs stay valid.
        assert open(cert, "rb").read() == first
