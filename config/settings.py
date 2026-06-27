import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-chunks-dev-key-change-in-production",
)

DEBUG = os.environ.get("DEBUG", "True").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "150"))
CHUNK_OVERLAP_SENTENCES = int(os.environ.get("CHUNK_OVERLAP_SENTENCES", "1"))
SEMANTIC_CHUNK_THRESHOLD = float(os.environ.get("SEMANTIC_CHUNK_THRESHOLD", "0.80"))
SEMANTIC_MAX_MERGED_TOKENS = int(os.environ.get("SEMANTIC_MAX_MERGED_TOKENS", "400"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "gpt-4o-mini")
LOOKBACK_MODEL = os.environ.get("LOOKBACK_MODEL", "") or EXTRACTION_MODEL
LOOKBACK_MAX_CHUNKS = int(os.environ.get("LOOKBACK_MAX_CHUNKS", "2"))
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "120"))
BULK_UPLOAD_MAX_FILES = int(os.environ.get("BULK_UPLOAD_MAX_FILES", "25"))
EXTRACT_CHUNK_WORKERS = int(os.environ.get("EXTRACT_CHUNK_WORKERS", "4"))
CLAIM_PROPOSITION_DEDUP_THRESHOLD = float(
    os.environ.get("CLAIM_PROPOSITION_DEDUP_THRESHOLD", "0.93")
)
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
RETRIEVAL_SIMILARITY_THRESHOLD = float(os.environ.get("RETRIEVAL_SIMILARITY_THRESHOLD", "0.70"))
RETRIEVAL_PHRASE_SIMILARITY_THRESHOLD = float(
    os.environ.get("RETRIEVAL_PHRASE_SIMILARITY_THRESHOLD", "0.85")
)
RETRIEVAL_LAYER_LIMIT = int(os.environ.get("RETRIEVAL_LAYER_LIMIT", "20"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.postgres",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "apps.transcripts",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

database_url = os.environ.get("DATABASE_URL", "")
db_name = os.environ.get("DB_NAME", "")
db_user = os.environ.get("DB_USER", "")
db_password = os.environ.get("DB_PASSWORD", "")
db_host = os.environ.get("DB_HOST", "")
db_port = os.environ.get("DB_PORT", "5432")

if database_url:
    parsed = urlparse(database_url)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": parsed.path.lstrip("/"),
            "USER": parsed.username,
            "PASSWORD": parsed.password,
            "HOST": parsed.hostname,
            "PORT": parsed.port or "5432",
        }
    }
elif db_name:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": db_name,
            "USER": db_user,
            "PASSWORD": db_password,
            "HOST": db_host,
            "PORT": db_port,
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "apps.transcripts": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
