#!/usr/bin/env bash
set -euo pipefail

SYNTHEA_DIR="data/synthea"
OUTPUT_DIR="data/synthea_output"
JAR_URL="https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"
JAR_PATH="${SYNTHEA_DIR}/synthea-with-dependencies.jar"

mkdir -p "${SYNTHEA_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${JAR_PATH}" ]; then
    echo "Downloading Synthea JAR..."
    curl -L -o "${JAR_PATH}" "${JAR_URL}"
fi

# Generate 20 patients in Massachusetts (default), FHIR R4 output as transaction bundles
java -jar "${JAR_PATH}" \
    -p 20 \
    --exporter.fhir.export true \
    --exporter.fhir.transaction_bundle true \
    --exporter.baseDirectory "${OUTPUT_DIR}" \
    Massachusetts

echo "Done. Bundles in ${OUTPUT_DIR}/fhir/"