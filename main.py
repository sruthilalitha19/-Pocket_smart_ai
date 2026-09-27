"""
PocketSmart AI - budget and recommendation web application.

FastAPI + Jinja2 + SQLite, with Gemini powering the three planners and a
deterministic local engine used whenever the AI service is unavailable.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

import database
import dependencies as deps
import security
from config import settings
from services import recommendation

app = FastAPI(title=settings.app_name, version="1.0.0")

settings.ensure_directories()
app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

templates = Jinja2Templates(directory=str(settings.templates_dir))

# Options offered by the planner forms, sourced from the catalogue.
from services import catalog  # noqa: E402  (after app setup to avoid cycles)

ROOMS = [
    "Living Room", "Bedroom", "Kitchen", "Bathroom",
    "Dining Area", "Study", "Balcony",
]
PARTY_TYPES = ["Birthday", "Wedding", "Corporate", "Anniversary", "Baby Shower"]
VENUE_TYPES = [
    "Home", "Residential hall booking", "Party lawn", "Conference hall",
]
PARTY_NEEDS = ["Venue", "Catering", "Entertainment", "Decoration", "Contingency"]
OCCASIONS = ["Birthday", "Wedding", "Corporate", "Anniversary", "Party", "Casual"]


@app.on_event("startup")
def on_startup() -> None:
    database.init_db()


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
def render(request: Request, name: str, context: dict[str, Any] | None = None,
          status_code: int = 200):
    """Render a template with the globals every page needs."""
    user = deps.get_current_user_optional(request)
    base: dict[str, Any] = {
        "request": request,
        "current_user": user,
        "app_name": settings.app_name,
        "ai_enabled": settings.ai_enabled,
        "gemini_model": settings.gemini_model,
    }
    base.update(context or {})
    # Newer Starlette expects (request, name, context).
    return templates.TemplateResponse(request, name, base, status_code=status_code)


def _require_login(request: Request):
    """Return the user, or a redirect response when signed out."""
    user = deps.get_current_user_optional(request)
    if user is None:
        return RedirectResponse(url="/login", status_code=303)
    return user



# ---------------------------------------------------------------------------
# Authentication pages
# ---------------------------------------------------------------------------
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if deps.get_current_user_optional(request) is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    return render(request, "register.html", {"error": None, "form": {}})


@app.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    values = {
        "username": username.strip(),
        "email": email.strip(),
        "confirm_password": confirm_password,
    }

    def fail(message: str):
        return render(
            request,
            "register.html",
            {"error": message, "form": values},
            status_code=400,
        )

    if len(username.strip()) < 3:
        return fail("Username must be at least 3 characters.")
    if not username.strip().replace("_", "").isalnum():
        return fail("Username may only contain letters, numbers and underscores.")
    if "@" not in email or "." not in email.split("@")[-1]:
        return fail("Enter a valid email address.")
    if len(password) < 6:
        return fail("Password must be at least 6 characters.")
    if password != confirm_password:
        return fail("Passwords do not match.")

    if database.get_user_by_username(username.strip()) is not None:
        return fail("That username is already taken.")
    if database.get_user_by_email(email.strip()) is not None:
        return fail("An account with that email already exists.")

    try:
        user_id = database.create_user(
            username.strip(), email.strip(), security.hash_password(password)
        )
    except sqlite3.IntegrityError:
        return fail("That username or email is already registered.")

    response = RedirectResponse(url="/dashboard", status_code=303)
    deps.set_session_cookie(response, user_id)
    return response


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if deps.get_current_user_optional(request) is not None:
        return RedirectResponse(url="/dashboard", status_code=303)
    return render(request, "login.html", {"error": None, "form": {}})


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    values = {"username": username.strip()}

    def fail(message: str, status_code: int = 400):
        return render(
            request,
            "login.html",
            {"error": message, "form": values},
            status_code=status_code,
        )

    if not username.strip() or not password:
        return fail("Enter your username and password.")

    row = database.get_user_by_username(username.strip())
    if row is None or not security.verify_password(password, row["password_hash"]):
        return fail("Incorrect username or password.", status_code=401)

    response = RedirectResponse(url="/dashboard", status_code=303)
    deps.set_session_cookie(response, row["id"])
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    deps.clear_session_cookie(response)
    return response



# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    user = deps.get_current_user_optional(request)
    recent: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    if user is not None:
        recent = database.list_recommendations(user["id"], limit=3)
        stats = database.dashboard_stats(user["id"])
    return render(
        request,
        "index.html",
        {"recent": recent, "stats": stats},
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return render(
        request,
        "dashboard.html",
        {
            "stats": database.dashboard_stats(user["id"]),
            "recent": database.list_recommendations(user["id"], limit=5),
        },
    )


@app.get("/planner/home", response_class=HTMLResponse)
def home_planner_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return render(
        request,
        "planner_home.html",
        {
            "rooms": ROOMS,
            "error": None,
            "plan": None,
            "form": {"budget": "", "lights": 1, "ceiling_fans": 1,
                     "furniture": 1, "dining_tables": 0,
                     "rooms": ["Living Room"], "requirements": ""},
        },
    )


@app.get("/planner/party", response_class=HTMLResponse)
def party_planner_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return render(
        request,
        "planner_party.html",
        {
            "party_types": PARTY_TYPES,
            "venue_types": VENUE_TYPES,
            "needs": PARTY_NEEDS,
            "error": None,
            "plan": None,
            "form": {"budget": "", "guests": 30, "party_type": "Birthday",
                     "venue_type": "Home", "needs": ["Catering"],
                     "requirements": ""},
        },
    )


@app.get("/planner/jewelry", response_class=HTMLResponse)
def jewelry_planner_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return render(
        request,
        "planner_jewelry.html",
        {
            "occasions": OCCASIONS,
            "error": None,
            "plan": None,
            "form": {"budget": "", "occasion": "Birthday",
                     "style_preferences": "", "outfit_image": None},
        },
    )


# ---------------------------------------------------------------------------
# Planner submissions
# ---------------------------------------------------------------------------
def _persist(user, category: str, req: dict[str, Any], plan: dict[str, Any],
             image_name: str | None = None) -> int:
    """Store a generated plan in the user's history."""
    return database.save_recommendation(
        user_id=user["id"],
        category=category,
        budget=plan["budget"],
        allocated=plan["allocated"],
        remaining=plan["remaining"],
        summary=plan.get("title", ""),
        inputs={**req, "outfit_image": image_name},
        result=plan,
        has_image=bool(image_name),
    )


async def _plan(fn, payload: dict[str, Any], image_bytes: bytes | None = None):
    """
    Run a blocking recommendation call off the event loop.

    The planners are ``async def`` routes (the jewelry one awaits an upload),
    so calling Gemini directly would stall every other request while it waits.
    """
    if image_bytes is None:
        return await run_in_threadpool(fn, payload)
    return await run_in_threadpool(fn, payload, image_bytes)


@app.post("/planner/home", response_class=HTMLResponse)
async def home_planner_submit(
    request: Request,
    budget: str = Form(...),
    lights: str = Form("1"),
    ceiling_fans: str = Form("1"),
    furniture: str = Form("1"),
    dining_tables: str = Form("0"),
    rooms: list[str] = Form([]),
    requirements: str = Form(""),
):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user

    budget_value = deps.as_float(budget, 0.0, 0.0)
    selected = [r for r in deps.checkbox_list(rooms) if r in ROOMS] or ["Living Room"]
    form = {
        "budget": budget,
        "lights": lights,
        "ceiling_fans": ceiling_fans,
        "furniture": furniture,
        "dining_tables": dining_tables,
        "rooms": selected,
        "requirements": requirements,
    }

    def fail(message: str):
        return render(
            request, "planner_home.html",
            {"rooms": ROOMS, "error": message, "plan": None, "form": form},
            status_code=400,
        )

    if budget_value <= 0:
        return fail("Enter a budget greater than zero.")

    payload = {
        "budget": budget_value,
        "lights": deps.as_int(lights, 1, 0),
        "ceiling_fans": deps.as_int(ceiling_fans, 1, 0),
        "furniture": deps.as_int(furniture, 1, 0),
        "dining_tables": deps.as_int(dining_tables, 0, 0),
        "rooms": selected,
        "requirements": requirements.strip(),
    }
    if not any(payload[k] for k in ("lights", "ceiling_fans", "furniture", "dining_tables")):
        return fail("Request at least one light, fan or furniture item.")

    plan = await _plan(recommendation.recommend_home, payload)
    _persist(user, "home", payload, plan)
    return render(
        request, "planner_home.html",
        {"rooms": ROOMS, "error": None, "plan": plan, "form": form},
    )


@app.post("/planner/party", response_class=HTMLResponse)
async def party_planner_submit(
    request: Request,
    budget: str = Form(...),
    guests: str = Form(...),
    party_type: str = Form("Birthday"),
    venue_type: str = Form("Home"),
    needs: list[str] = Form([]),
    requirements: str = Form(""),
):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user

    budget_value = deps.as_float(budget, 0.0, 0.0)
    guest_count = deps.as_int(guests, 0, 0)
    selected = [n for n in deps.checkbox_list(needs) if n in PARTY_NEEDS] or ["Catering"]
    form = {
        "budget": budget,
        "guests": guests,
        "party_type": party_type,
        "venue_type": venue_type,
        "needs": selected,
        "requirements": requirements,
    }

    def fail(message: str):
        return render(
            request, "planner_party.html",
            {"party_types": PARTY_TYPES, "venue_types": VENUE_TYPES,
             "needs": PARTY_NEEDS, "error": message, "plan": None, "form": form},
            status_code=400,
        )

    if budget_value <= 0:
        return fail("Enter a budget greater than zero.")
    if guest_count < 1:
        return fail("Enter how many guests are coming.")
    if party_type not in PARTY_TYPES:
        return fail("Choose a valid party type.")

    payload = {
        "budget": budget_value,
        "guests": guest_count,
        "party_type": party_type,
        "venue_type": venue_type if venue_type in VENUE_TYPES else "Home",
        "needs": selected,
        "requirements": requirements.strip(),
    }

    plan = await _plan(recommendation.recommend_party, payload)
    _persist(user, "party", payload, plan)
    return render(
        request, "planner_party.html",
        {"party_types": PARTY_TYPES, "venue_types": VENUE_TYPES,
         "needs": PARTY_NEEDS, "error": None, "plan": plan, "form": form},
    )


@app.post("/planner/jewelry", response_class=HTMLResponse)
async def jewelry_planner_submit(
    request: Request,
    budget: str = Form(...),
    occasion: str = Form("Birthday"),
    style_preferences: str = Form(""),
    outfit_image: UploadFile | None = None,
):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user

    budget_value = deps.as_float(budget, 0.0, 0.0)
    form = {
        "budget": budget,
        "occasion": occasion,
        "style_preferences": style_preferences,
        "outfit_image": None,
    }

    def fail(message: str, status_code: int = 400):
        return render(
            request, "planner_jewelry.html",
            {"occasions": OCCASIONS, "error": message, "plan": None, "form": form},
            status_code=status_code,
        )

    if budget_value <= 0:
        return fail("Enter a budget greater than zero.")
    if occasion not in OCCASIONS:
        return fail("Choose a valid occasion.")

    try:
        uploaded = await deps.save_outfit_image(outfit_image)
    except HTTPException as exc:
        return fail(exc.detail, status_code=exc.status_code)

    image_name = uploaded[0] if uploaded else None
    image_bytes = uploaded[1] if uploaded else None
    form["outfit_image"] = image_name

    payload = {
        "budget": budget_value,
        "occasion": occasion,
        "style_preferences": style_preferences.strip(),
    }

    plan = await _plan(recommendation.recommend_jewelry, payload, image_bytes)
    _persist(user, "jewelry", payload, plan, image_name=image_name)
    return render(
        request, "planner_jewelry.html",
        {"occasions": OCCASIONS, "error": None, "plan": plan, "form": form},
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request, category: str | None = None):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    if category not in {None, "home", "party", "jewelry"}:
        category = None
    return render(
        request,
        "history.html",
        {
            "entries": database.list_recommendations(user["id"], category=category),
            "active_category": category,
            "stats": database.dashboard_stats(user["id"]),
        },
    )


@app.get("/history/{reco_id}", response_class=HTMLResponse)
def history_detail(request: Request, reco_id: int):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    entry = database.get_recommendation(reco_id, user["id"])
    if entry is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    return render(
        request,
        "history_detail.html",
        {"entry": entry, "plan": entry.get("result", {})},
    )


@app.post("/history/{reco_id}/delete")
def history_delete(request: Request, reco_id: int):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    database.delete_recommendation(reco_id, user["id"])
    return RedirectResponse(url="/history", status_code=303)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
def _api_user(user=Depends(deps.get_bearer_user)):
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Provide a bearer token from POST /api/login",
        )
    return user


@app.get("/api/session")
def api_session(user=Depends(deps.get_bearer_user)):
    """Report who the caller is (works signed in or out)."""
    if user is None:
        return {"authenticated": False, "user_id": None,
                "username": None, "login_status": "not_logged_in"}
    return {"authenticated": True, "user_id": user["id"],
            "username": user["username"], "login_status": "logged_in"}


@app.post("/api/register")
def api_register(payload: dict):
    username = str(payload.get("username", "")).strip()
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))
    confirm = str(payload.get("confirm_password", password))

    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if "@" not in email:
        raise HTTPException(400, "Enter a valid email address.")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    if password != confirm:
        raise HTTPException(400, "Passwords do not match.")
    if database.get_user_by_username(username):
        raise HTTPException(409, "That username is already taken.")
    if database.get_user_by_email(email):
        raise HTTPException(409, "An account with that email already exists.")

    user_id = database.create_user(
        username, email, security.hash_password(password)
    )
    return {
        "user": {"id": user_id, "username": username, "email": email},
        "access_token": security.create_access_token(user_id),
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


@app.post("/api/login")
def api_login(payload: dict):
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    row = database.get_user_by_username(username) if username else None
    if row is None or not security.verify_password(password, row["password_hash"]):
        raise HTTPException(401, "Incorrect username or password")
    return {
        "user": {"id": row["id"], "username": row["username"], "email": row["email"]},
        "access_token": security.create_access_token(row["id"]),
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


def _api_budget(payload: dict) -> float:
    try:
        budget = float(payload.get("budget", 0))
    except (TypeError, ValueError):
        raise HTTPException(422, "budget must be a number")
    if budget <= 0:
        raise HTTPException(422, "budget must be greater than zero")
    return budget


@app.post("/api/recommend/home")
def api_recommend_home(payload: dict, user=Depends(_api_user)):
    req = {
        "budget": _api_budget(payload),
        "lights": int(payload.get("lights", 1) or 0),
        "ceiling_fans": int(payload.get("ceiling_fans", 1) or 0),
        "furniture": int(payload.get("furniture", 1) or 0),
        "dining_tables": int(payload.get("dining_tables", 0) or 0),
        "rooms": payload.get("rooms") or ["Living Room"],
        "requirements": str(payload.get("requirements", "")),
    }
    plan = recommendation.recommend_home(req)
    plan_id = _persist(user, "home", req, plan)
    return {"ok": True, "category": "home",
            "engine": plan["engine"], "plan_id": plan_id, "data": plan}


@app.post("/api/recommend/party")
def api_recommend_party(payload: dict, user=Depends(_api_user)):
    try:
        guests = int(payload.get("guests", 0))
    except (TypeError, ValueError):
        raise HTTPException(422, "guests must be a number")
    if guests < 1:
        raise HTTPException(422, "guests must be at least 1")

    req = {
        "budget": _api_budget(payload),
        "guests": guests,
        "party_type": str(payload.get("party_type", "Birthday")),
        "venue_type": str(payload.get("venue_type", "Home")),
        "needs": payload.get("needs") or ["Catering"],
        "requirements": str(payload.get("requirements", "")),
    }
    plan = recommendation.recommend_party(req)
    plan_id = _persist(user, "party", req, plan)
    return {"ok": True, "category": "party",
            "engine": plan["engine"], "plan_id": plan_id, "data": plan}


@app.post("/api/recommend/jewelry")
def api_recommend_jewelry(payload: dict, user=Depends(_api_user)):
    req = {
        "budget": _api_budget(payload),
        "occasion": str(payload.get("occasion", "Birthday")),
        "style_preferences": str(payload.get("style_preferences", "")),
    }
    plan = recommendation.recommend_jewelry(req, None)
    plan_id = _persist(user, "jewelry", req, plan)
    return {"ok": True, "category": "jewelry",
            "engine": plan["engine"], "plan_id": plan_id, "data": plan}


@app.get("/api/history")
def api_history(user=Depends(_api_user), category: str | None = None):
    return {
        "entries": database.list_recommendations(user["id"], category=category),
        "stats": database.dashboard_stats(user["id"]),
    }


@app.delete("/api/history/{reco_id}")
def api_history_delete(reco_id: int, user=Depends(_api_user)):
    if not database.delete_recommendation(reco_id, user["id"]):
        raise HTTPException(404, "Plan not found")
    return {"ok": True, "deleted": reco_id}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
def _wants_json(request: Request) -> bool:
    return request.url.path.startswith("/api/")


@app.exception_handler(404)
async def not_found(request: Request, exc):
    if _wants_json(request):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return render(request, "404.html", {"detail": "Page not found"},
                  status_code=404)


@app.exception_handler(500)
async def server_error(request: Request, exc):  # pragma: no cover
    if _wants_json(request):
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
    return render(request, "500.html", {}, status_code=500)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    """Render HTML errors for pages, JSON for the API."""
    if _wants_json(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    if exc.status_code == 404:
        return render(request, "404.html", {"detail": "Page not found"},
                      status_code=404)
    if exc.status_code == 401:
        return RedirectResponse(url="/login", status_code=303)
    return render(request, "error.html", {"detail": exc.detail},
                  status_code=exc.status_code)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )

