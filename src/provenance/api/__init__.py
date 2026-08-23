"""HTTP API for the Text Provenance Engine.

Local-first REST API exposing the provenance analysis engine with pluggable
persistence (SQLite or PostgreSQL).  Requires the optional ``api`` extra:

    pip install -e '.[api]'

Start the server (default SQLite):

    uvicorn provenance.api.app:app --host 0.0.0.0 --port 8000

With PostgreSQL:

    export DATABASE_URL='postgresql://user:pass@localhost/provenance'
    uvicorn provenance.api.app:app --host 0.0.0.0 --port 8000
"""
