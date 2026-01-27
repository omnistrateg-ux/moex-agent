#!/bin/bash
#
# MOEX Agent — Деплой в Yandex Cloud Serverless Containers
#
# Использование:
#   chmod +x deploy/yandex-deploy.sh
#   ./deploy/yandex-deploy.sh
#

set -e  # Exit on error

# ===== КОНФИГУРАЦИЯ =====
FOLDER_ID="${FOLDER_ID:-b1g58hgssbaevs2m924t}"
REGISTRY_ID="${REGISTRY_ID:-crp8htjr9iulov5vtf0l}"
CONTAINER_NAME="${CONTAINER_NAME:-moex-agent}"
SERVICE_ACCOUNT_ID="${SERVICE_ACCOUNT_ID:-ajeopktlqun8vakd75m6}"

IMAGE_NAME="moex-agent"
IMAGE_TAG="${IMAGE_TAG:-latest}"

MEMORY="${MEMORY:-1024Mi}"
CORES="${CORES:-1}"
CORE_FRACTION="${CORE_FRACTION:-100}"
EXECUTION_TIMEOUT="${EXECUTION_TIMEOUT:-300s}"
CONCURRENCY="${CONCURRENCY:-1}"

DATABASE_URL="${DATABASE_URL:-postgresql://moexagent:MoexAgent2026!Secure@rc1b-5421i5tvv2p060mk.mdb.yandexcloud.net:6432/moexdb?sslmode=require}"

# Telegram из config.yaml
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-8429791732:AAFRkt_xV42XV0L4U5IS5DK6Dkstn_XROAk}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-120171956}"

# ===== ЦВЕТА =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ===== MAIN =====
main() {
    echo ""
    echo "======================================"
    echo "  MOEX Agent — Yandex Cloud Deploy"
    echo "======================================"
    echo ""

    # Переходим в корень проекта
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_DIR"

    # Проверки
    log_info "Проверка зависимостей..."
    command -v docker &>/dev/null || { log_error "Docker не найден"; exit 1; }
    command -v yc &>/dev/null || command -v ~/yandex-cloud/bin/yc &>/dev/null || { log_error "yc CLI не найден"; exit 1; }

    # Используем yc из PATH или из home
    if command -v yc &>/dev/null; then
        YC="yc"
    else
        YC="$HOME/yandex-cloud/bin/yc"
    fi

    log_success "Docker: $(docker --version | cut -d' ' -f3)"
    log_success "yc CLI: $($YC version 2>/dev/null | head -1)"

    # Полный путь к образу
    FULL_IMAGE="cr.yandex/${REGISTRY_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

    # Сборка
    log_info "Сборка Docker образа..."
    docker build -f deploy/Dockerfile -t "$FULL_IMAGE" .
    log_success "Образ собран: $FULL_IMAGE"

    # Авторизация в registry
    log_info "Авторизация в Yandex Container Registry..."
    $YC container registry configure-docker
    log_success "Docker авторизован"

    # Push
    log_info "Push образа в registry..."
    docker push "$FULL_IMAGE"
    log_success "Образ загружен"

    # Создание контейнера если нет
    log_info "Проверка Serverless Container..."
    if ! $YC serverless container get --name "$CONTAINER_NAME" &>/dev/null; then
        log_info "Создание контейнера: $CONTAINER_NAME"
        $YC serverless container create --name "$CONTAINER_NAME" --folder-id "$FOLDER_ID"
    fi

    # Деплой ревизии
    log_info "Деплой новой ревизии..."
    $YC serverless container revision deploy \
        --container-name "$CONTAINER_NAME" \
        --image "$FULL_IMAGE" \
        --service-account-id "$SERVICE_ACCOUNT_ID" \
        --memory "$MEMORY" \
        --cores "$CORES" \
        --core-fraction "$CORE_FRACTION" \
        --execution-timeout "$EXECUTION_TIMEOUT" \
        --concurrency "$CONCURRENCY" \
        --environment "DATABASE_URL=${DATABASE_URL},PORT=8080,TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}"

    # Публичный доступ
    log_info "Настройка публичного доступа..."
    CONTAINER_ID=$($YC serverless container get --name "$CONTAINER_NAME" --format json | jq -r '.id')
    $YC serverless container allow-unauthenticated-invoke --id "$CONTAINER_ID" 2>/dev/null || true

    # Результат
    CONTAINER_URL=$($YC serverless container get --name "$CONTAINER_NAME" --format json | jq -r '.url')

    echo ""
    echo "======================================"
    log_success "ДЕПЛОЙ ЗАВЕРШЁН!"
    echo "======================================"
    echo ""
    echo "Container URL: $CONTAINER_URL"
    echo ""
    echo "Dashboard: ${CONTAINER_URL}/"
    echo "Health:    ${CONTAINER_URL}/api/health"
    echo "API:       ${CONTAINER_URL}/api/status"
    echo ""
}

main "$@"
