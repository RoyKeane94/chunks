import json
import random

from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db.models import Count, Prefetch, Q
from django.http import HttpResponse
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.text import slugify

from .layer_status import (
    build_unresolved_lookback_export,
    get_extraction_summary,
    list_episode_layer_status,
    resolve_unresolved_item,
    delete_unresolved_item,
)
from .models import AtomicPhrase, Chunk, Claim, Episode, Proposition
from .services import (
    find_episode_by_content_hash,
    get_retrieval_config,
    hash_transcript_content,
    ingest_bulk_files,
    ingest_episode,
    read_uploaded_transcript,
    retrieve_similar_chunks,
)
from .templatetags.transcript_extras import build_retrieve_snippet


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

GLASS_MOSAIC_COUNT = 32
GLASS_SNIPPET_MAX = 72


def _truncate_snippet(text: str, max_len: int = GLASS_SNIPPET_MAX) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return f"{cut}…"


def glass_mosaic_items(count: int = GLASS_MOSAIC_COUNT) -> list[dict]:
    """Sample real extraction rows for the landing-page glass mosaic."""
    per_layer = (count + 2) // 3
    items = []
    layer_models = [
        (Proposition, "prop"),
        (Claim, "claim"),
        (AtomicPhrase, "phrase"),
    ]
    for model, layer in layer_models:
        rows = (
            model.objects.select_related("chunk__episode")
            .only(
                "content",
                "chunk__episode_id",
                "chunk__episode__title",
                "chunk__episode__guest",
            )
            .order_by("?")[:per_layer]
        )
        for row in rows:
            episode = row.chunk.episode
            items.append({
                "layer": layer,
                "content": _truncate_snippet(row.content),
                "episode_id": episode.id,
                "episode_title": episode.title,
                "episode_guest": episode.guest,
            })
    random.shuffle(items)
    return items[:count]


def landing(request):
    featured_episodes = list(
        Episode.objects.only(*EPISODE_LIST_FIELDS)
        .annotate(chunk_count=Count("chunks"))
        .order_by("-created_at")[:6]
    )
    stats = {
        "episode_count": Episode.objects.count(),
        "chunk_count": Chunk.objects.count(),
    }
    glass_items = glass_mosaic_items()
    return render(
        request,
        "transcripts/landing.html",
        {
            "featured_episodes": featured_episodes,
            "stats": stats,
            "glass_items": glass_items,
        },
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


def extraction_status(request):
    summary = get_extraction_summary()
    filter_key = request.GET.get("filter", "").strip()
    episodes = list_episode_layer_status()

    if filter_key == "fully_layered":
        episodes = [e for e in episodes if e["status"] == "fully_layered"]
    elif filter_key == "partial":
        episodes = [e for e in episodes if e["status"] == "partial"]
    elif filter_key == "not_started":
        episodes = [e for e in episodes if e["status"] == "not_started"]
    elif filter_key == "unresolved":
        episodes = [e for e in episodes if e["unresolved_count"] > 0]

    return render(
        request,
        "transcripts/extraction_status.html",
        {
            "summary": summary,
            "episodes": episodes,
            "filter_key": filter_key,
        },
    )


def unresolved_lookback(request):
    ep_id = None
    episode_id = request.GET.get("episode_id") or request.POST.get("episode_id")
    if episode_id:
        try:
            ep_id = int(episode_id)
        except ValueError:
            pass

    if request.method == "POST":
        layer = request.POST.get("layer", "").strip()
        try:
            item_id = int(request.POST.get("item_id", ""))
        except ValueError:
            item_id = None
        content = request.POST.get("content", "")
        action = request.POST.get("action", "resolve").strip()

        if item_id is None:
            messages.error(request, "Invalid item.")
        elif action == "delete":
            ok, err = delete_unresolved_item(layer, item_id)
            if ok:
                messages.success(request, f"Deleted {layer} #{item_id}.")
            else:
                messages.error(request, err)
        else:
            ok, err = resolve_unresolved_item(layer, item_id, content)
            if ok:
                messages.success(request, f"Marked {layer} #{item_id} as resolved.")
            else:
                messages.error(request, err)

        redirect_url = reverse("unresolved_lookback")
        if ep_id:
            redirect_url = f"{redirect_url}?episode_id={ep_id}"
        if item_id and action != "delete":
            redirect_url = f"{redirect_url}#item-{item_id}"
        return redirect(redirect_url)

    export = build_unresolved_lookback_export(ep_id)
    episode_options = list_episode_layer_status()
    episode_options = [e for e in episode_options if e["unresolved_count"] > 0]

    return render(
        request,
        "transcripts/unresolved_lookback.html",
        {
            "export": export,
            "episode_id": ep_id,
            "episode_options": episode_options,
        },
    )


def unresolved_lookback_download(request):
    episode_id = request.GET.get("episode_id")
    ep_id = None
    if episode_id:
        try:
            ep_id = int(episode_id)
        except ValueError:
            pass

    export = build_unresolved_lookback_export(ep_id)
    stamp = export["generated_at"].replace(":", "").replace("-", "")
    suffix = f"_ep{ep_id}" if ep_id else ""
    filename = f"unresolved_lookback{suffix}_{stamp}.json"
    response = HttpResponse(
        json.dumps(export, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


def retrieve(request):
    query = request.GET.get("q", "").strip()
    results = []
    error = None

    if query:
        try:
            raw_results = retrieve_similar_chunks(query)
            results = []
            for result in raw_results:
                snippet = build_retrieve_snippet(
                    result["chunk"].content,
                    result.get("highlight_start"),
                    result.get("highlight_end"),
                    result.get("matched_content"),
                    result.get("matched_layer"),
                )
                results.append({**result, "snippet": snippet})
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"Retrieval failed: {exc}"

    threshold, phrase_threshold, top_k, _layer_limit = get_retrieval_config()
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
            proposition_count=Count("propositions", distinct=True),
            claim_count=Count("claims", distinct=True),
            phrase_count=Count("atomic_phrases", distinct=True),
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
            transcript_file = form.cleaned_data["transcript_file"]
            raw_text = read_uploaded_transcript(transcript_file)
            content_hash = hash_transcript_content(raw_text)
            existing = find_episode_by_content_hash(content_hash)
            if existing:
                form.add_error(
                    None,
                    f"This transcript is already in the library as «{existing.title}» "
                    f"(episode {existing.id}).",
                )
            else:
                episode = Episode.objects.create(
                    title=form.cleaned_data["title"],
                    guest=form.cleaned_data["guest"],
                    date=form.cleaned_data.get("date"),
                    pdf_file=transcript_file,
                    content_hash=content_hash,
                )
                ingest_report = ingest_episode(episode.id, raw_text=raw_text)
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
