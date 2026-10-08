#!/usr/bin/env bash
set -euo pipefail

echo "================================"
echo "HuggingFace LFS Downloader"
echo "================================"

for cmd in git git-lfs aria2c sha256sum awk sed cut basename dirname mktemp; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Error: ${cmd} not installed"
        exit 1
    fi
done

REPO_INPUT="${1:-}"

if [ -z "${REPO_INPUT}" ]; then
    echo "Usage:"
    echo "  $0 <hf_repo_url_or_id>"
    exit 1
fi

if [[ "${REPO_INPUT}" =~ ^https?:// ]]; then
    REPO_INPUT="${REPO_INPUT%/}"
    REPO_INPUT="${REPO_INPUT%.git}"

    GIT_URL="${REPO_INPUT}.git"

    REPO_PATH=$(echo "${REPO_INPUT}" \
        | sed -E 's#https?://[^/]+/##')

    BASE_DOMAIN=$(echo "${REPO_INPUT}" \
        | awk -F'//' '{print $2}' \
        | awk -F'/' '{print $1}')
else
    BASE_DOMAIN="huggingface.co"
    REPO_PATH="${REPO_INPUT%.git}"
    GIT_URL="https://${BASE_DOMAIN}/${REPO_PATH}.git"
fi

MODEL_DIR=$(basename "${REPO_PATH}")

if [ -z "${MODEL_DIR}" ] || [ "${MODEL_DIR}" = "/" ]; then
    echo "Invalid repository"
    exit 1
fi

echo
echo "==============================="
echo "Repository : ${REPO_PATH}"
echo "Domain     : ${BASE_DOMAIN}"
echo "Directory  : ${MODEL_DIR}"
echo "Git URL    : ${GIT_URL}"
echo "==============================="
echo

if [ -d "${MODEL_DIR}/.git" ]; then

    echo "Existing repository found."

    cd "${MODEL_DIR}"

    echo "Initializing Git LFS..."
    git lfs install --local

    echo "Updating repository metadata..."
    GIT_LFS_SKIP_SMUDGE=1 git fetch origin

    CURRENT_BRANCH=$(git symbolic-ref --quiet --short HEAD || true)

    if [ -n "${CURRENT_BRANCH}" ]; then
        if git show-ref --verify --quiet \
            "refs/remotes/origin/${CURRENT_BRANCH}"; then

            git reset --hard "origin/${CURRENT_BRANCH}"

        fi
    fi

else

    if [ -d "${MODEL_DIR}" ]; then
        echo "Removing incomplete directory..."
        rm -rf "${MODEL_DIR}"
    fi

    echo "Cloning repository metadata..."

    GIT_LFS_SKIP_SMUDGE=1 \
    git clone "${GIT_URL}" "${MODEL_DIR}"

    cd "${MODEL_DIR}"

    echo "Initializing Git LFS..."
    git lfs install --local

fi

REVISION=$(git rev-parse HEAD)

echo
echo "Revision:"
echo "${REVISION}"
echo

echo "Collecting LFS objects..."

LFS_LIST_FILE=$(mktemp)

trap 'rm -f "${LFS_LIST_FILE}"' EXIT

git lfs ls-files --long > "${LFS_LIST_FILE}"

if [ ! -s "${LFS_LIST_FILE}" ]; then
    echo
    echo "No LFS files."
    exit 0
fi

echo
cat "${LFS_LIST_FILE}"
echo

download_lfs_file()
{
    local HASH="$1"
    local FILE="$2"

    local DIR
    local FILENAME
    local URL
    local LOCAL_HASH
    local ARIA2_STATUS

    DIR=$(dirname "${FILE}")
    FILENAME=$(basename "${FILE}")

    mkdir -p "${DIR}"

    URL="https://${BASE_DOMAIN}/${REPO_PATH}/resolve/${REVISION}/${FILE}"

    echo
    echo "--------------------------------"
    echo "File:"
    echo "${FILE}"
    echo
    echo "SHA256:"
    echo "${HASH}"
    echo
    echo "URL:"
    echo "${URL}"
    echo "--------------------------------"

    while true
    do
        echo
        echo "Downloading ${FILE}..."

        set +e

        aria2c \
            --check-certificate=false \
            --continue=true \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --max-connection-per-server=16 \
            --split=16 \
            --min-split-size=1M \
            --max-tries=5 \
            --retry-wait=5 \
            --timeout=60 \
            --connect-timeout=30 \
            --file-allocation=none \
            -d "${DIR}" \
            -o "${FILENAME}" \
            "${URL}"

        ARIA2_STATUS=$?

        set -e

        if [ "${ARIA2_STATUS}" -ne 0 ]; then
            echo
            echo "aria2c failed with exit code ${ARIA2_STATUS}"

            if [ ! -f "${FILE}" ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
                continue
            fi
        fi

        if [ ! -f "${FILE}" ]; then
            echo
            echo "File not found:"
            echo "${FILE}"

            echo "Retrying in 5 seconds..."
            sleep 5
            continue
        fi

        echo
        echo "Verifying SHA256..."

        LOCAL_HASH=$(sha256sum "${FILE}" | awk '{print $1}')

        if [ "${LOCAL_HASH}" = "${HASH}" ]; then
            echo
            echo "HASH OK:"
            echo "${FILE}"
            break
        fi

        echo
        echo "HASH FAILED"
        echo "Expected: ${HASH}"
        echo "Actual:   ${LOCAL_HASH}"

        rm -f "${FILE}"

        echo "Retrying in 5 seconds..."
        sleep 5
    done
}

while IFS= read -r line
do
    HASH=$(echo "${line}" | awk '{print $1}')
    FILE=$(echo "${line}" | cut -d' ' -f3-)

    if [ -z "${FILE}" ]; then
        continue
    fi

    if ! [[ "${HASH}" =~ ^[0-9a-fA-F]{64}$ ]]; then
        echo
        echo "Invalid SHA256:"
        echo "${HASH}"
        echo "Line:"
        echo "${line}"
        exit 1
    fi

    echo
    echo "Parsed LFS object:"
    echo "  SHA  : ${HASH}"
    echo "  FILE : ${FILE}"

    download_lfs_file \
        "${HASH}" \
        "${FILE}"

done < "${LFS_LIST_FILE}"

echo
echo "Checking LFS objects..."

git lfs checkout

echo
echo "Final LFS status:"

git lfs ls-files

echo
echo "================================"
echo "Download finished successfully"
echo "================================"

echo
echo "Model directory:"
pwd
