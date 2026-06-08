from django.urls import path

from . import views

urlpatterns = [
    path("", views.episode_list, name="episode_list"),
    path("episodes/<int:episode_id>/", views.episode_detail, name="episode_detail"),
    path("episodes/<int:episode_id>/download/", views.episode_download_json, name="episode_download"),
    path("episodes/<int:episode_id>/delete/", views.episode_delete, name="episode_delete"),
    path("upload/", views.upload, name="upload"),
    path("upload/bulk/", views.bulk_upload, name="bulk_upload"),
    path("upload/bulk/confirm/", views.bulk_upload_confirm, name="bulk_upload_confirm"),
    path("upload/confirm/<int:episode_id>/", views.upload_confirm, name="upload_confirm"),
]
