from django.apps import AppConfig


class PredictConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predict'

from django.apps import AppConfig


class YourAppNameConfig(AppConfig):
    name = 'predict'

    def ready(self):
        import predict.signals  # Import signals here
