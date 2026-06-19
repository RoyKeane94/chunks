from django.contrib import admin

from .models import AtomicPhrase, Chunk, Claim, Episode, Proposition


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
        "embedded_at",
        "extraction_model",
        "extracted_at",
        "created_at",
    )


@admin.register(Proposition)
class PropositionAdmin(admin.ModelAdmin):
    list_display = ("chunk", "content", "embedding_model", "embedded_at", "extraction_model", "created_at")


@admin.register(Claim)
class ClaimAdmin(admin.ModelAdmin):
    list_display = ("chunk", "content", "embedding_model", "embedded_at", "extraction_model", "created_at")


@admin.register(AtomicPhrase)
class AtomicPhraseAdmin(admin.ModelAdmin):
    list_display = ("chunk", "content", "embedding_model", "embedded_at", "extraction_model", "created_at")
