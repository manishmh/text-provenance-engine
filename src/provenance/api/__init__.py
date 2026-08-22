"""HTTP API for the Text Provenance Engine.

Local-first REST API exposing the provenance analysis engine with SQLite
persistence.  Requires the optional ``api`` extra:

    pip install -e '.[api]'

Start the server:

    uvicorn provenance.api.app:app --host 0.0.0.0 --port 8000
"""
