#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# Docker image blob downloader
# Registry -> skopeo dir format
#
# Features:
#   - multi arch support
#   - digest verification
#   - safe atomic download
#   - retry on network failure
# =====================================================


# ==================== 默认参数 ====================

TEMP_DIR="./vllm_temp_dir"

IMAGE_URL="docker.io/vllm/vllm-omni:latest"

TARGET_PLATFORM=""


# ==================== 参数解析 ====================

while getopts "d:i:p:h" opt; do
case $opt in

d)
    TEMP_DIR="$OPTARG"
    ;;

i)
    IMAGE_URL="$OPTARG"
    ;;

p)
    TARGET_PLATFORM="$OPTARG"
    ;;

h|*)
cat <<EOF

Usage:

bash $0 \
    -d <download_dir> \
    -i <image> \
    -p <platform>


Example:

bash $0 \
    -d ./vllm_temp_dir \
    -i docker.io/vllm/vllm-omni:latest \
    -p amd64


EOF
exit 0
;;

esac
done



# ==================== 架构识别 ====================

if [ -z "${TARGET_PLATFORM}" ]; then

    ARCH=$(uname -m)

    case ${ARCH} in

        x86_64)
            TARGET_PLATFORM="amd64"
            ;;

        aarch64|arm64)
            TARGET_PLATFORM="arm64"
            ;;

        *)
            echo "Unknown arch ${ARCH}"
            exit 1
            ;;

    esac

fi



echo
echo "=============================="
echo "IMAGE : ${IMAGE_URL}"
echo "DIR   : ${TEMP_DIR}"
echo "ARCH  : ${TARGET_PLATFORM}"
echo "=============================="
echo



mkdir -p "${TEMP_DIR}"



# ==================== 解析镜像 ====================


IMAGE_PURE=${IMAGE_URL#docker://}


REGISTRY=$(echo "${IMAGE_PURE}" | cut -d'/' -f1)


if [[ "${IMAGE_PURE}" != */* ]]; then

    IMAGE_REPO="library/${IMAGE_PURE}"

else

    IMAGE_REPO=$(echo "${IMAGE_PURE}" | cut -d'/' -f2-)

fi


TAG="${IMAGE_REPO##*:}"

IMAGE_REPO="${IMAGE_REPO%:*}"



if [[ "${REGISTRY}" == "docker.io" ]]; then

    API_DOMAIN="registry-1.docker.io"

    AUTH_URL="https://auth.docker.io/token?service=registry.docker.io&scope=repository:${IMAGE_REPO}:pull"

else

    API_DOMAIN="${REGISTRY}"

    AUTH_URL="https://${REGISTRY}/v2/auth?service=${REGISTRY}&scope=repository:${IMAGE_REPO}:pull"

fi



echo "Repository:"
echo "${IMAGE_REPO}"



# ==================== 获取 token ====================


get_token()
{
    curl -ks "${AUTH_URL}" | jq -r '.token'
}



TOKEN=$(get_token)


if [[ "${TOKEN}" == "null" || -z "${TOKEN}" ]]; then

    echo "Failed to get token"

    exit 1

fi



# ==================== 获取 manifest index ====================


echo
echo "==== Get manifest index ===="


INDEX_JSON=$(

skopeo inspect \
--tls-verify=false \
--raw \
docker://${IMAGE_PURE}

)



if echo "${INDEX_JSON}" | jq -e '.manifests' >/dev/null 2>&1
then

    MANIFEST_DIGEST=$(

    echo "${INDEX_JSON}" |

    jq -r \
    --arg ARCH "${TARGET_PLATFORM}" \
    '

    .manifests[]
    |
    select(.platform.architecture==$ARCH)
    |
    .digest

    '

    )


    if [ -z "${MANIFEST_DIGEST}" ]; then

        echo "Cannot find platform ${TARGET_PLATFORM}"

        exit 1

    fi


else

    MANIFEST_DIGEST="${TAG}"

fi



echo "Manifest:"
echo "${MANIFEST_DIGEST}"




# ==================== 下载 manifest ====================


echo
echo "==== Download manifest ===="


curl -ks \
-H "Authorization: Bearer ${TOKEN}" \
-H "Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" \
"https://${API_DOMAIN}/v2/${IMAGE_REPO}/manifests/${MANIFEST_DIGEST}" \
-o "${TEMP_DIR}/manifest.json"



jq -e '.layers' "${TEMP_DIR}/manifest.json" >/dev/null



# ==================== 下载 blob ====================


download_blob()
{

local digest=$1


local sha=${digest#sha256:}


local target="${TEMP_DIR}/${sha}"

local tmp="${target}.tmp"



echo
echo "--------------------------------"
echo "Blob ${sha:0:12}"



while true
do


    if [ -f "${target}" ]; then

        CURRENT=$(sha256sum "${target}" | awk '{print $1}')

        if [ "${CURRENT}" == "${sha}" ]; then

            echo "Already OK"

            return

        else

            echo "Bad existing blob remove"

            rm -f "${target}"

        fi

    fi



    TOKEN=$(get_token)



    echo "Downloading..."



    rm -f "${tmp}"



    if curl -ksL \
        -H "Authorization: Bearer ${TOKEN}" \
        -o "${tmp}" \
        "https://${API_DOMAIN}/v2/${IMAGE_REPO}/blobs/${digest}"
    then


        SUM=$(sha256sum "${tmp}" | awk '{print $1}')


        if [ "${SUM}" == "${sha}" ]; then


            mv "${tmp}" "${target}"

            echo "OK"

            return


        else

            echo
            echo "Digest mismatch"
            echo "Expected:"
            echo "${sha}"
            echo "Got:"
            echo "${SUM}"


            rm -f "${tmp}"

        fi


    fi



    echo "Retry after 5 seconds..."

    sleep 5


done

}



echo
echo "==== Download blobs ===="



CONFIG=$(jq -r '.config.digest' "${TEMP_DIR}/manifest.json")


download_blob "${CONFIG}"



jq -r '.layers[].digest' "${TEMP_DIR}/manifest.json" |

while read layer
do

    download_blob "${layer}"

done




# ==================== skopeo dir 标志 ====================


echo "1.1" > "${TEMP_DIR}/version"



echo
echo "================================="
echo "All blobs downloaded successfully"
echo "Directory:"
echo "${TEMP_DIR}"
echo "================================="

