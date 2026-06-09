from django.contrib import admin

from .models import AtomicClaim, Chunk, Episode, Proposition


@admin.register(Episode)
class EpisodeAdmin(admin.ModelAdmin):
    list_display = ("title", "guest", "date", "created_at")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = (
        "episode",
        "chunk_index",
        "token_estimate",
        "embedding_model",
        "extraction_model",
        "extracted_at",
        "created_at",
    )


@admin.register(AtomicClaim)
class AtomicClaimAdmin(admin.ModelAdmin):
    list_display = ("chunk", "ac_content", "embedding_model", "created_at")


@admin.register(Proposition)
class PropositionAdmin(admin.ModelAdmin):
    list_display = ("chunk", "prop_content", "embedding_model", "created_at")
