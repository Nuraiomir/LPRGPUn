#!/usr/bin/env bash
# Starts the service, creating the two things it cannot ship with: a TLS
# certificate and an access key.
#
# Neither may live in the image. A certificate baked into an image is the same
# certificate on every machine that pulls it, and a key baked into an image is
# not a key. Both are written to /certs, which is a volume, so they survive a
# restart and an image rebuild.
#
# The browser needs HTTPS before it will give a page access to the camera, and
# a self-signed certificate is enough for that: the browser warns once, the
# warning is accepted once per browser, and the camera works. A certificate
# from a real authority is a question for whoever runs this in production.
set -euo pipefail

CERT_DIR="${LPR_CERT_DIR:-/certs}"
CERT="${CERT_DIR}/server.pem"
KEY="${CERT_DIR}/server.key"
KEYS_FILE="${CERT_DIR}/api_keys.txt"
CN="${LPR_CERT_CN:-localhost}"

mkdir -p "${CERT_DIR}"

if [ ! -f "${CERT}" ] || [ ! -f "${KEY}" ]; then
    echo "сертификата нет, создаю самоподписанный на имя ${CN}"
    echo "если заходить будешь по IP, задай LPR_CERT_CN=<этот IP> и удали ${CERT_DIR}/server.*"
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
        -subj "/CN=${CN}" -keyout "${KEY}" -out "${CERT}" 2>/dev/null
    chmod 600 "${KEY}"
fi

# A key from the environment wins: that is how a real deployment passes one.
# The generated file is the fallback so that a container started with nothing
# is still closed rather than open to anyone who can reach the port.
if [ -z "${LPR_API_KEYS:-}" ]; then
    if [ ! -f "${KEYS_FILE}" ]; then
        python -c 'import secrets; print(secrets.token_urlsafe(32))' > "${KEYS_FILE}"
        chmod 600 "${KEYS_FILE}"
        echo "=============================================================="
        echo "СОЗДАН КЛЮЧ ДОСТУПА. Он нужен, чтобы открыть страницу сканера:"
        cat "${KEYS_FILE}"
        echo "Лежит в ${KEYS_FILE}, печатается только при создании."
        echo "=============================================================="
    fi
    set -- --api-keys-file "${KEYS_FILE}" "$@"
fi

exec python app/lpr_api_server.py --cert "${CERT}" --key "${KEY}" "$@"
