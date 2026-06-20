import json

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Count, Prefetch, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.text import slugify

from .models import AtomicPhrase, Claim, Episode, Proposition
from .services import get_retrieval_config, ingest_bulk_files, ingest_episode, retrieve_similar_chunks


def validate_transcript_file(file):
    if not file.name.lower().endswith((".pdf", ".txt")):
        raise ValidationError("Upload a PDF or TXT file.")


class UploadForm(forms.Form):
    title = forms.CharField(max_length=255)
    guest = forms.CharField(max_length=255)
    date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    transcript_file = forms.FileField(
        validators=[validate_transcript_file],
        widget=forms.FileInput(attrs={"accept": ".pdf,.txt"}),
    )


EPISODES_PER_PAGE = 20
EPISODE_LIST_FIELDS = ("id", "title", "guest", "date", "created_at")
LAYER_LIST_FIELDS = ("id", "chunk_id", "content", "source_text")
CHUNK_DETAIL_FIELDS = (
    "id",
    "episode_id",
    "content",
    "chunk_index",
    "token_estimate",
    "extracted_at",
)
CHUNK_DOWNLOAD_FIELDS = (
    "chunk_index",
    "content",
    "token_estimate",
    "embedding_model",
    "embedded_at",
    "embedding",
    "created_at",
)


def episode_list(request):
    query = request.GET.get("q", "").strip()
    episodes = (
        Episode.objects.only(*EPISODE_LIST_FIELDS)
        .annotate(
            chunk_count=Count("chunks"),
            layered_chunk_count=Count(
                "chunks",
                filter=Q(chunks__extracted_at__isnull=False),
            ),
        )
        .order_by("-created_at")
    )
    if query:
        episodes = episodes.filter(Q(title__icontains=query) | Q(guest__icontains=query))
    page_obj = Paginator(episodes, EPISODES_PER_PAGE).get_page(request.GET.get("page"))
    return render(
        request,
        "transcripts/episode_list.html",
        {"page_obj": page_obj, "query": query},
    )


def retrieve(request):
    query = request.GET.get("q", "").strip()
    results = []
    error = None

    if query:
        try:
            results = retrieve_similar_chunks(query)
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Retrieval failed: {exc}"

    threshold, phrase_threshold, top_k = get_retrieval_config()
    return render(
        request,
        "transcripts/retrieve.html",
        {
            "query": query,
            "results": results,
            "threshold": threshold,
            "phrase_threshold": phrase_threshold,
            "top_k": top_k,
            "error": error,
        },
    )


def episode_detail(request, episode_id):
    layer_prefetches = (
        Prefetch(
            "propositions",
            queryset=Proposition.objects.only(*LAYER_LIST_FIELDS).defer("embedding"),
        ),
        Prefetch(
            "claims",
            queryset=Claim.objects.only(*LAYER_LIST_FIELDS).defer("embedding"),
        ),
        Prefetch(
            "atomic_phrases",
            queryset=AtomicPhrase.objects.only(*LAYER_LIST_FIELDS).defer("embedding"),
        ),
    )
    episode = get_object_or_404(
        Episode.objects.annotate(chunk_count=Count("chunks")).only(
            "id", "title", "guest", "date"
        ),
        pk=episode_id,
    )
    chunks = (
        episode.chunks.defer("embedding")
        .only(*CHUNK_DETAIL_FIELDS)
        .annotate(
            proposition_count=Count("propositions"),
            claim_count=Count("claims"),
            phrase_count=Count("atomic_phrases"),
        )
        .prefetch_related(*layer_prefetches)
        .order_by("chunk_index")
    )
    return render(
        request,
        "transcripts/episode_detail.html",
        {"episode": episode, "chunks": chunks},
    )


def episode_download_json(request, episode_id):
    episode = get_object_or_404(
        Episode.objects.only("id", "title", "guest", "date", "created_at"),
        pk=episode_id,
    )
    data = {
        "episode": {
            "id": episode.id,
            "title": episode.title,
            "guest": episode.guest,
            "date": episode.date.isoformat() if episode.date else None,
            "created_at": episode.created_at.isoformat(),
        },
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_estimate": chunk.token_estimate,
                "embedding_model": chunk.embedding_model,
                "embedded_at": chunk.embedded_at.isoformat() if chunk.embedded_at else None,
                "embedding": [float(x) for x in chunk.embedding],
                "created_at": chunk.created_at.isoformat(),
            }
            for chunk in episode.chunks.only(*CHUNK_DOWNLOAD_FIELDS).order_by("chunk_index").iterator(
                chunk_size=100
            )
        ],
    }
    filename = f"{slugify(episode.title) or 'episode'}-{episode.id}.json"
    response = HttpResponse(json.dumps(data, indent=2), content_type="application/json")
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def episode_delete(request, episode_id):
    episode = get_object_or_404(Episode, pk=episode_id)
    if request.method == "POST":
        if episode.pdf_file:
            episode.pdf_file.delete(save=False)
        episode.delete()
    return redirect("episode_list")


def upload(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            episode = Episode.objects.create(
                title=form.cleaned_data["title"],
                guest=form.cleaned_data["guest"],
                date=form.cleaned_data.get("date"),
                pdf_file=form.cleaned_data["transcript_file"],
            )
            ingest_report = ingest_episode(episode.id)
            request.session["ingest_report"] = ingest_report
            return redirect("upload_confirm", episode_id=episode.id)
    else:
        form = UploadForm()

    return render(request, "transcripts/upload.html", {"form": form})


def upload_confirm(request, episode_id):
    episode = get_object_or_404(
        Episode.objects.only("id", "title", "guest", "date"),
        pk=episode_id,
    )
    ingest_report = request.session.pop("ingest_report", None)
    return render(
        request,
        "transcripts/upload_confirm.html",
        {"episode": episode, "report": ingest_report},
    )


def bulk_upload(request):
    max_files = settings.BULK_UPLOAD_MAX_FILES
    error = None

    if request.method == "POST":
        files = request.FILES.getlist("transcript_files")
        if not files:
            error = "Select at least one .txt file."
        elif len(files) > max_files:
            error = f"Maximum {max_files} files per batch. Upload in smaller groups."
        else:
            try:
                results = ingest_bulk_files(files)
                request.session["bulk_upload_results"] = results
                return redirect("bulk_upload_confirm")
            except ValueError as exc:
                error = str(exc)

    return render(
        request,
        "transcripts/bulk_upload.html",
        {"max_files": max_files, "error": error},
    )


def bulk_upload_confirm(request):
    results = request.session.pop("bulk_upload_results", None)
    return render(
        request,
        "transcripts/bulk_upload_confirm.html",
        {"results": results},
    )
