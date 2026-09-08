#!/usr/bin/env bash
# Regenerate the dev HTTPS cert for whatever LAN IP this machine currently has.
# RUN THIS AT THE VENUE: the hackathon Wi-Fi will hand out a different IP, and a
# cert whose SAN does not list it makes iOS Safari refuse the page outright.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p certs

IPS=$(node -e 'const o=require("os").networkInterfaces();const a=[];for(const l of Object.values(o))for(const n of l||[])if(n.family==="IPv4"&&!n.internal)a.push(n.address);console.log(a.join(","))')
SAN="DNS:localhost, IP:127.0.0.1"
for ip in ${IPS//,/ }; do SAN="$SAN, IP:$ip"; done
echo "SAN -> $SAN"

cat > certs/openssl.cnf <<CNF
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = aiswarm.local
[v3]
subjectAltName = $SAN
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
CNF

OPENSSL_CONF=certs/openssl.cnf openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem -days 90 -config certs/openssl.cnf 2>/dev/null
openssl x509 -in certs/cert.pem -noout -ext subjectAltName
echo "cert regenerated - restart the server"
