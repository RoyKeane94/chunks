from django.db import models
from pgvector.django import VectorField


class Episode(models.Model):
    title = models.CharField(max_length=255)
    guest = models.CharField(max_length=255)
    date = models.DateField(null=True, blank=True)
    pdf_file = models.FileField(upload_to="pdfs/")
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
    extraction_model = models.CharField(max_length=100, blank=True)
    extracted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["chunk_index"]

    def __str__(self):
        return f"{self.episode.title} — chunk {self.chunk_index}"


class AtomicClaim(models.Model):
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="atomic_claims")
    ac_content = models.TextField()
    embedding = VectorField(dimensions=1536)
    embedding_model = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.chunk} — claim: {self.ac_content[:60]}"


class Proposition(models.Model):
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name="propositions")
    prop_content = models.TextField()
    embedding = VectorField(dimensions=1536)
    embedding_model = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.chunk} — proposition: {self.prop_content[:60]}"
