from django.contrib import admin

from .models import Chunk, Episode


@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = ("title", "guest", "date", "created_at")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("episode", "chunk_index", "token_estimate", "embedding_model", "created_at")
