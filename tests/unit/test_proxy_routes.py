#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Calls between two agents through a proxy that puts itself on the route.

Once a proxy Record-Routes, every request either side sends inside the
call (the ACK, a hold, the hang-up) goes to the proxy's route address
and not to the peer. The forms below are the ones proxies emit; the
``sips:`` ones are what Flexisip (sip.linphone.org) sends, and a
``sips:`` Contact is what OpenAI's SIP endpoint answers with. Each test
watches the other end to see that the request arrived.
"""

import asyncio
import itertools
import shutil
import subprocess

import pytest

native = pytest.importorskip("baresip._native")

from sip_proxy import SipProxy

from baresip import Account, CallState, Config, Event, Runtime, UserAgent

# A closed runtime's SIP socket lingers in-process; sequential tests
# each take a fresh port.
_PORTS = itertools.count(5385)

# name: (the caller's transport, Record-Route entries, replacement Contact)
#
# The proxy reaches the answering agent over UDP whatever the caller
# uses. Record-Route entries are listed top first: the answering side
# takes the first as its next hop, the caller the last. A proxy that
# bridges two transports inserts one entry per side; one that insists on
# TLS inserts a single ``sips:`` entry for both.
FORMS = {
    "sip over udp": ("udp", ("<sip:127.0.0.1:{udp};lr>",), None),
    "sip, one entry per side": (
        "tls",
        ("<sip:127.0.0.1:{udp};lr>", "<sip:127.0.0.1:{tls};transport=tls;lr>"),
        None,
    ),
    "sips": ("tls", ("<sips:127.0.0.1:{tls};lr>",), None),
    "sips with a hostname": ("tls", ("<sips:localhost:{tls};lr>",), None),
    "sips, caller on udp": ("udp", ("<sips:127.0.0.1:{tls};lr>",), None),
    "sips for the caller, sip for the answerer": (
        "tls",
        ("<sip:127.0.0.1:{udp};lr>", "<sips:127.0.0.1:{tls};lr>"),
        None,
    ),
    "sips contact, no route": ("tls", (), "sips:{user}@127.0.0.1:{tls}"),
}
# The stack refuses a request whose next hop is a sips: URI.
SIPS = pytest.mark.xfail(strict=True, reason="libre does not send to a sips: next hop")
CASES = [pytest.param(form, marks=SIPS) if "sips" in form else form for form in FORMS]


@pytest.fixture(scope="session")
def certificate(tmp_path_factory):
    """A self-signed certificate for the proxy, naming 127.0.0.1 and localhost."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    folder = tmp_path_factory.mktemp("proxy-tls")
    cert, key = folder / "cert.pem", folder / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2"]
        + ["-keyout", str(key), "-out", str(cert), "-subj", "/CN=127.0.0.1"]
        + ["-addext", "subjectAltName=IP:127.0.0.1,DNS:localhost"],
        check=True,
        capture_output=True,
    )
    return str(cert), str(key)


@pytest.fixture
async def routed(certificate, monkeypatch):
    """Place a call through the proxy in a given form: ``await routed(form)``."""
    cert, key = certificate
    monkeypatch.setenv("SIP_CA_FILE", cert)
    port = next(_PORTS)
    own = f"127.0.0.1:{port}"
    runtime = Runtime()
    await runtime.start(Config(net_interface="127.0.0.1", extra_config_text=f"sip_listen {own}\n"))
    proxies: list[SipProxy] = []

    async def connect(form: str, target: str | None = None, caller_transport: str | None = None):
        transport, record_route, contact = FORMS[form]
        transport = caller_transport or transport
        proxy = SipProxy(("127.0.0.1", port), cert, key, record_route, contact)
        await proxy.start()
        proxies.append(proxy)
        caller = await UserAgent.create(
            runtime,
            Account(user="alice", password="", domain=own, reg_interval=0, transport=transport),
        )
        callee = await UserAgent.create(
            runtime, Account(user="bob", password="", domain=own, reg_interval=0)
        )
        incoming: asyncio.Queue = asyncio.Queue()
        callee.on_incoming(incoming.put_nowait)
        if target is None:
            proxy_port = proxy.tls_port if transport == "tls" else proxy.udp_port
            target = f"sip:bob@127.0.0.1:{proxy_port};transport={transport}"
        placed = await caller.dial(target.format(udp=proxy.udp_port, tls=proxy.tls_port))
        received = await asyncio.wait_for(incoming.get(), 5)
        await received.answer()
        return placed, received, proxy

    yield connect
    await runtime.close()
    for proxy in proxies:
        await proxy.close()


def closed(call) -> asyncio.Future:
    """A future resolved when the call is closed."""
    future = asyncio.get_running_loop().create_future()
    if call.state is CallState.CLOSED:
        future.set_result(None)
        return future

    def listener(event):
        if event.event is Event.CALL_CLOSED and not future.done():
            future.set_result(None)

    call.on(listener)
    return future


@pytest.mark.parametrize("form", CASES)
async def test_the_call_completes_and_the_callers_hangup_arrives(routed, form):
    placed, received, _proxy = await routed(form)

    # Established on the answering side means the caller's ACK arrived.
    await placed.wait_established(timeout=5)
    await received.wait_established(timeout=5)

    gone = closed(received)
    await placed.hangup()
    await asyncio.wait_for(gone, 2)


@pytest.mark.parametrize("form", CASES)
async def test_the_answering_sides_hangup_arrives(routed, form):
    placed, received, _proxy = await routed(form)
    await placed.wait_established(timeout=5)
    await received.wait_established(timeout=5)

    gone = closed(placed)
    await received.hangup()
    await asyncio.wait_for(gone, 2)


@pytest.mark.parametrize("form", CASES)
async def test_a_hold_from_the_answering_side_arrives(routed, form):
    placed, received, _proxy = await routed(form)
    await placed.wait_established(timeout=5)
    await received.wait_established(timeout=5)
    await asyncio.sleep(0.5)  # let the stack's own follow-up re-INVITE settle

    await received.hold()
    for _ in range(40):
        if placed.remote_on_hold:
            break
        await asyncio.sleep(0.05)
    assert placed.remote_on_hold

    await received.resume()
    for _ in range(40):
        if not placed.remote_on_hold:
            break
        await asyncio.sleep(0.05)
    assert not placed.remote_on_hold


@SIPS
async def test_requests_on_a_sips_route_travel_over_tls(routed):
    placed, received, proxy = await routed("sips, caller on udp")
    await placed.wait_established(timeout=5)
    await received.wait_established(timeout=5)
    gone = closed(placed)
    await received.hangup()
    await asyncio.wait_for(gone, 2)

    arrived_on = dict(proxy.requests)
    assert arrived_on["INVITE"] == "UDP"  # dialled over UDP, before any route existed
    assert arrived_on["ACK"] == "TLS"
    assert arrived_on["BYE"] == "TLS"


@SIPS
async def test_a_sips_target_is_dialled_over_tls_from_a_udp_account(routed):
    placed, received, proxy = await routed(
        "sips", target="sips:bob@127.0.0.1:{tls}", caller_transport="udp"
    )

    await placed.wait_established(timeout=5)
    await received.wait_established(timeout=5)
    assert proxy.requests[0] == ("INVITE", "TLS")
