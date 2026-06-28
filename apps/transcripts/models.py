from django.db import models
from pgvector.django import HnswIndex, VectorField


class Episode(models.Model):
    title = models.CharField(max_length=255)
    guest = models.CharField(max_length=255)
    date = models.DateField(null=True, blank=True)
    pdf_file = models.FileField(upload_to="pdfs/")
    content_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Chunk(models.Model):
    episode = models.ForeignKey(Episode, on_delete=models.CASCADE, related_name="chunks")
    content = models.TextField()
    chunk_index = models.IntegerField()
    token_estimate = models.IntegerField()
    embedding = VectorField(dimensions=1536)
    embedding_model = models.CharField(max_length=100)
    embedded_at = models.DateTimeField(null=True, blank=True)
    extraction_model = models.CharField(max_length=100, blank=True)
    extracted_at = models.DateTimeField(null=True, blank=True)
    lookback_completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["chunk_index"]
        indexes = [
            HnswIndex(
                fields=["embedding"],
                name="chunk_embedding_hnsw_idx",
                opclasses=["vector_cosine_ops"],
                m=8,
                ef_construction=32,
            ),
        ]

    def __str__(self):
        return f"{self.episode.title} — chunk {self.chunk_index}"


class Proposition(models.Model):
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="propositions")
    content = models.TextField()
    source_text = models.TextField()
    start_char = models.IntegerField()
    end_char = models.IntegerField()
    embedding = VectorField(dimensions=1536)
    embedding_model = models.CharField(max_length=64, default="text-embedding-3-small")
    embedded_at = models.DateTimeField(null=True, blank=True)
    extraction_model = models.CharField(max_length=64, default="gpt-4o-mini")
    needs_lookback = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                fields=["embedding"],
                name="proposition_embedding_hnsw_idx",
                opclasses=["vector_cosine_ops"],
                m=8,
                ef_construction=32,
            ),
        ]

    def __str__(self):
        return f"{self.chunk} — proposition: {self.content[:60]}"


class Claim(models.Model):
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="claims")
    content = models.TextField()
    source_text = models.TextField()
    start_char = models.IntegerField()
    end_char = models.IntegerField()
    embedding = VectorField(dimensions=1536)
    embedding_model = models.CharField(max_length=64, default="text-embedding-3-small")
    embedded_at = models.DateTimeField(null=True, blank=True)
    extraction_model = models.CharField(max_length=64, default="gpt-4o-mini")
    needs_lookback = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                fields=["embedding"],
                name="claim_embedding_hnsw_idx",
                opclasses=["vector_cosine_ops"],
                m=8,
                ef_construction=32,
            ),
        ]

    def __str__(self):
        return f"{self.chunk} — claim: {self.content[:60]}"


class AtomicPhrase(models.Model):
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="atomic_phrases")
    content = models.TextField()
    source_text = models.TextField()
    start_char = models.IntegerField()
    end_char = models.IntegerField()
    embedding = VectorField(dimensions=1536)
    embedding_model = models.CharField(max_length=64, default="text-embedding-3-small")
    embedded_at = models.DateTimeField(null=True, blank=True)
    extraction_model = models.CharField(max_length=64, default="gpt-4o-mini")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            HnswIndex(
                fields=["embedding"],
                name="phrase_embedding_hnsw_idx",
                opclasses=["vector_cosine_ops"],
                m=8,
                ef_construction=32,
            ),
        ]

    def __str__(self):
        return f"{self.chunk} — phrase: {self.content[:60]}"
