@echo off
echo Iniciando Celery Worker para Moonchart Backend...
echo ADVERTENCIA: Ejecutando en modo "-P solo" para prevenir congelamiento en Windows.
celery -A app.worker.celery_app worker --loglevel=info -P solo
pause
