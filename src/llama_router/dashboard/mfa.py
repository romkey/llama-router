"""Local dashboard password + WebAuthn, enrollment and account security.

Protocol verification uses py_webauthn, not custom cryptography:
https://duo-labs.github.io/py_webauthn/
"""

from __future__ import annotations

import json
import secrets
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from webauthn import (
    base64url_to_bytes,
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import bytes_to_base64url
from webauthn.helpers.exceptions import WebAuthnException
from webauthn.helpers.parse_authentication_credential_json import (
    parse_authentication_credential_json,
)
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    UserVerificationRequirement,
)

from ..config import settings
from . import deps
from .auth_core import AUTH_COOKIE, unsign_session
from .mfa_store import FLOW_TTL, Credential, Flow, LocalAccount, MfaStore, digest
from .routes import (
    _cookie_secure,
    _set_session_cookie,
    dash_page_ctx,
    templates,
)

router = APIRouter()
PENDING_COOKIE = "lr_second_factor"


class CeremonyStart(BaseModel):
    name: str = Field(default="YubiKey", min_length=1, max_length=100)


class CeremonyFinish(BaseModel):
    token: str = Field(min_length=1, max_length=200)
    credential: str = Field(min_length=1, max_length=65536)


class RecoveryAttempt(BaseModel):
    code: str = Field(min_length=1, max_length=100)


def _private(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


def _origin() -> tuple[str, str]:
    origin = settings.dashboard_webauthn_origin
    host = urlsplit(origin).hostname
    if not origin or not host:
        raise HTTPException(
            status_code=503,
            detail="Configure LLAMA_ROUTER_DASHBOARD_WEBAUTHN_ORIGIN to use WebAuthn",
        )
    return origin, host


def _csrf(request: Request) -> None:
    # Cross-origin pages cannot set this header without a CORS preflight.
    # These routes do not grant CORS access. Also reject mismatched Origins.
    origin, _ = _origin()
    if (
        request.headers.get("x-llama-router-security") != "1"
        or request.headers.get("origin", origin) != origin
    ):
        raise HTTPException(
            status_code=403, detail="Security requests must come from this dashboard"
        )


async def _valid_flow(token: str, purpose: str) -> tuple[Flow, LocalAccount]:
    store = MfaStore(deps.get_db())
    flow = await store.flow(token, purpose)
    account = await store.account(flow.user_id) if flow else None
    if flow is None or account is None or flow.state["version"] != str(account.version):
        raise HTTPException(status_code=401, detail="Sign-in expired. Sign in again.")
    return flow, account


async def _pending(request: Request) -> tuple[Flow, LocalAccount]:
    return await _valid_flow(request.cookies.get(PENDING_COOKIE, ""), "pending")


async def _session_account(request: Request) -> LocalAccount:
    secret = await deps.get_dashboard_session_secret()
    claims = unsign_session(request.cookies.get(AUTH_COOKIE, ""), secret)
    account = await MfaStore(deps.get_db()).account(claims.user_id) if claims else None
    if (
        account is None
        or claims is None
        or claims.version != account.version
        or claims.username != account.username
        or (account.enabled and not claims.mfa_verified)
    ):
        raise HTTPException(status_code=403, detail="Sign in with a local account")
    return account


async def _set_session(
    request: Request, response: Response, account: LocalAccount, *, verified: bool
) -> None:
    secret = await deps.get_dashboard_session_secret()
    _set_session_cookie(
        request,
        response,
        {
            "id": account.id,
            "username": account.username,
            "session_version": account.version,
        },
        secret,
        mfa_verified=verified,
    )
    response.delete_cookie(PENDING_COOKIE)
    await deps.get_db().touch_dashboard_user_login(account.id)
    _private(response)


async def password_login_response(
    request: Request, user_id: int, next_url: str, *, session_version: int
) -> Response:
    store = MfaStore(deps.get_db())
    account = await store.account(user_id)
    if account is None or account.version != session_version:
        raise HTTPException(status_code=401, detail="Invalid login")
    if (
        settings.dashboard_webauthn_origin
        and request.headers.get("origin", settings.dashboard_webauthn_origin)
        != settings.dashboard_webauthn_origin
    ):
        url = "/login?error=origin"
        if next_url != "/":
            url += "&next=" + quote(next_url, safe="")
        return _private(RedirectResponse(url, status_code=303))
    if not account.enabled and not settings.dashboard_require_second_factor:
        response = RedirectResponse(next_url, status_code=303)
        await _set_session(request, response, account, verified=False)
        return response
    token = await store.new_flow(
        account,
        "pending",
        next=next_url,
        mode="verify" if account.enabled else "enroll",
    )
    response = RedirectResponse("/login/second-factor", status_code=303)
    response.delete_cookie(AUTH_COOKIE)
    response.set_cookie(
        PENDING_COOKIE,
        token,
        httponly=True,
        samesite="strict",
        secure=_cookie_secure(request)
        or settings.dashboard_webauthn_origin.startswith("https://"),
        max_age=FLOW_TTL,
    )
    return _private(response)


@router.get("/login/second-factor")
async def second_factor_page(request: Request) -> Response:
    try:
        flow, _ = await _pending(request)
    except HTTPException:
        return _private(RedirectResponse("/login", status_code=303))
    return _private(
        templates.TemplateResponse(
            request,
            "second_factor.html",
            {"enrolling": flow.state["mode"] == "enroll", "dash_user": None},
        )
    )


@router.get("/profile")
@router.get("/account/security")
async def security_page(request: Request) -> Response:
    user = getattr(request.state, "dashboard_user", None)
    if user is None:
        raise HTTPException(status_code=403, detail="Sign in to view your account")
    credentials = []
    if not user.is_sso:
        account = await _session_account(request)
        credentials = await MfaStore(deps.get_db()).credentials(account.id)
    return _private(
        templates.TemplateResponse(
            request,
            "security.html",
            await dash_page_ctx(
                request,
                {
                    "credentials": credentials,
                    "webauthn_configured": bool(settings.dashboard_webauthn_origin),
                    "mfa_required": settings.dashboard_require_second_factor,
                },
            ),
        )
    )


async def _registration_account(request: Request) -> tuple[LocalAccount, str]:
    if request.url.path.startswith("/login/"):
        flow, account = await _pending(request)
        if flow.state["mode"] != "enroll" or account.enabled:
            raise HTTPException(
                status_code=403,
                detail="Complete second factor before adding an authenticator",
            )
        return account, flow.token
    return await _session_account(request), ""


@router.post("/account/security/register/options")
@router.post("/login/second-factor/register/options")
async def registration_options(request: Request, body: CeremonyStart) -> Response:
    _csrf(request)
    account, pending = await _registration_account(request)
    origin, rp_id = _origin()
    store = MfaStore(deps.get_db())
    credentials = await store.credentials(account.id)
    options = generate_registration_options(
        rp_id=rp_id,
        rp_name="llama-router",
        user_name=account.username,
        user_id=f"llama-router:{account.id}".encode(),
        exclude_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c.id))
            for c in credentials
        ],
        authenticator_selection=AuthenticatorSelectionCriteria(
            user_verification=UserVerificationRequirement.REQUIRED
        ),
    )
    token = await store.new_flow(
        account,
        "register",
        challenge=bytes_to_base64url(options.challenge),
        origin=origin,
        rp_id=rp_id,
        name=body.name.strip() or "YubiKey",
        pending=digest(pending) if pending else "",
    )
    return _private(
        JSONResponse({"token": token, "options": json.loads(options_to_json(options))})
    )


def _codes() -> list[str]:
    # 128 bits per recovery code; fast hashing is safe for random secrets.
    return [secrets.token_hex(16) for _ in range(10)]


@router.post("/account/security/register/verify")
@router.post("/login/second-factor/register/verify")
async def registration_verify(request: Request, body: CeremonyFinish) -> Response:
    _csrf(request)
    account, pending = await _registration_account(request)
    flow, owner = await _valid_flow(body.token, "register")
    if owner.id != account.id or flow.state["pending"] != (
        digest(pending) if pending else ""
    ):
        raise HTTPException(
            status_code=403, detail="Enrollment belongs to another sign-in"
        )
    try:
        verified = verify_registration_response(
            credential=body.credential,
            expected_challenge=base64url_to_bytes(flow.state["challenge"]),
            expected_rp_id=flow.state["rp_id"],
            expected_origin=flow.state["origin"],
            require_user_verification=True,
        )
    except (WebAuthnException, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Authenticator registration could not be verified"
        ) from exc
    codes = _codes() if not account.enabled else []
    credential = Credential(
        id=bytes_to_base64url(verified.credential_id),
        user_id=account.id,
        name=flow.state["name"],
        public_key=bytes_to_base64url(verified.credential_public_key),
        sign_count=verified.sign_count,
    )
    store = MfaStore(deps.get_db())
    try:
        saved = await store.register(flow, account, credential, codes)
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409, detail="This authenticator is already registered"
        ) from exc
    if not saved:
        raise HTTPException(
            status_code=401, detail="Enrollment expired. Sign in again."
        )
    updated = await store.account(account.id)
    if updated is None:
        raise HTTPException(status_code=401, detail="Sign in again")
    response = JSONResponse({"ok": True, "recovery_codes": codes, "next": "/profile"})
    await _set_session(request, response, updated, verified=True)
    return response


@router.post("/login/second-factor/authenticate/options")
async def authentication_options(request: Request) -> Response:
    _csrf(request)
    pending, account = await _pending(request)
    if not account.enabled:
        raise HTTPException(status_code=403, detail="Enroll an authenticator first")
    origin, rp_id = _origin()
    store = MfaStore(deps.get_db())
    credentials = await store.credentials(account.id)
    options = generate_authentication_options(
        rp_id=rp_id,
        allow_credentials=[
            PublicKeyCredentialDescriptor(id=base64url_to_bytes(c.id))
            for c in credentials
        ],
        user_verification=UserVerificationRequirement.REQUIRED,
    )
    token = await store.new_flow(
        account,
        "authenticate",
        challenge=bytes_to_base64url(options.challenge),
        origin=origin,
        rp_id=rp_id,
        pending=digest(pending.token),
    )
    return _private(
        JSONResponse({"token": token, "options": json.loads(options_to_json(options))})
    )


@router.post("/login/second-factor/authenticate/verify")
async def authentication_verify(request: Request, body: CeremonyFinish) -> Response:
    _csrf(request)
    pending, account = await _pending(request)
    flow, owner = await _valid_flow(body.token, "authenticate")
    if (
        owner.id != account.id
        or flow.state["pending"] != digest(pending.token)
        or not account.enabled
    ):
        raise HTTPException(
            status_code=403, detail="Challenge belongs to another sign-in"
        )
    store = MfaStore(deps.get_db())
    try:
        parsed = parse_authentication_credential_json(body.credential)
        credential = next(
            (c for c in await store.credentials(account.id) if c.id == parsed.id), None
        )
        if credential is None:
            raise ValueError("Unknown credential")
        verified = verify_authentication_response(
            credential=parsed,
            expected_challenge=base64url_to_bytes(flow.state["challenge"]),
            expected_rp_id=flow.state["rp_id"],
            expected_origin=flow.state["origin"],
            credential_public_key=base64url_to_bytes(credential.public_key),
            credential_current_sign_count=credential.sign_count,
            require_user_verification=True,
        )
    except (WebAuthnException, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Second factor could not be verified"
        ) from exc
    if not await store.finish_authentication(
        flow, account, credential=credential, sign_count=verified.new_sign_count
    ):
        raise HTTPException(status_code=401, detail="Challenge expired. Sign in again.")
    response = JSONResponse({"ok": True, "next": pending.state["next"]})
    await _set_session(request, response, account, verified=True)
    return response


@router.post("/login/second-factor/recovery")
async def recovery_login(request: Request, body: RecoveryAttempt) -> Response:
    _csrf(request)
    pending, account = await _pending(request)
    if not account.enabled or not await MfaStore(deps.get_db()).finish_authentication(
        pending, account, recovery_code=body.code
    ):
        raise HTTPException(
            status_code=401, detail="Invalid recovery code. Sign in again."
        )
    response = JSONResponse({"ok": True, "next": "/profile"})
    await _set_session(request, response, account, verified=True)
    return response


@router.post("/account/security/recovery-codes")
async def regenerate_codes(request: Request) -> Response:
    _csrf(request)
    account = await _session_account(request)
    if not account.enabled:
        raise HTTPException(status_code=400, detail="Register an authenticator first")
    codes = _codes()
    if not await MfaStore(deps.get_db()).replace_codes(account, codes):
        raise HTTPException(status_code=401, detail="Sign in again")
    return _private(JSONResponse({"recovery_codes": codes}))


@router.delete("/account/security/credentials/{credential_id}")
async def remove_credential(request: Request, credential_id: str) -> Response:
    _csrf(request)
    account = await _session_account(request)
    if not await MfaStore(deps.get_db()).delete_credential(account, credential_id):
        raise HTTPException(
            status_code=400,
            detail="Keep at least one authenticator; register a replacement first",
        )
    updated = await MfaStore(deps.get_db()).account(account.id)
    if updated is None:
        raise HTTPException(status_code=401, detail="Sign in again")
    response = JSONResponse({"ok": True})
    await _set_session(request, response, updated, verified=True)
    return response
