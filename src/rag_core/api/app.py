from fastapi import FastAPI

from rag_core.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="rag-core")
    app.include_router(router)
    return app
