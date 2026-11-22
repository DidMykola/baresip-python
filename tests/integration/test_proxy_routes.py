#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Calls between two agents through the bench's proxy, on each route form.

Kamailio (bench/conf/kamailio.cfg) Record-Routes every call, and the
port a call enters on picks the form. Once it does, everything either
side sends inside the call (the ACK, a hold, DTMF as INFO, a transfer
request, the hang-up) goes to the proxy's route address. Each test
watches the other end to see that it arrived. Run with:

    pytest -m bench tests/integration
"""

import asyncio
import contextlib
import pathlib

import pytest

pytest.importorskip("baresip._native")

from baresip import Account, AudioRestarted, BaresipError, CallState, Config, Event
from baresip.runtime import Runtime
from baresip.ua import UserAgent

pytestmark = pytest.mark.bench

BENCH_CERT = pathlib.Path(__file__).resolve().parents[2] / "bench" / "certs" / "cert.pem"
REGISTRAR = "127.0.0.1:15070"
# The route form, by the proxy port a call enters on and the transport to it.
FORMS = {
    "sip over udp": (15070, "udp"),
    "sip, one entry per side": (15071, "tls"),
    "sips": (15072, "tls"),
    "sips with a hostname": (15073, "tls"),
    "sips for the caller, sip for the answerer": (15074, "tls"),
}


@pytest.fixture
async def runtime(monkeypatch):
    monkeypatch.setenv("SIP_CA_FILE", str(BENCH_CERT))
    rt = Runtime()
    await rt.start(Config(net_interface="127.0.0.1", max_concurrent_calls=None))
    yield rt
    await rt.close()


@pytest.fixture
def routed(runtime):
    """Place a call through the proxy in a given form: ``await routed(form)``."""

    async def connect(form: str, **callee_fields):
        port, transport = FORMS[form]
        callee = await UserAgent.create(
            runtime, Account(user="1004", password="", domain=REGISTRAR, **callee_fields)
        )
        await callee.register()
        caller = await UserAgent.create(
            runtime,
            Account(
                user="1003", password="", domain=REGISTRAR, reg_interval=0, transport=transport
            ),
        )
        incoming: asyncio.Queue = asyncio.Queue()
        callee.on_incoming(incoming.put_nowait)
        placed = await caller.dial(f"sip:1004@127.0.0.1:{port};transport={transport}")
        received = await asyncio.wait_for(incoming.get(), 10)
        await received.answer()
        # Established on the answering side means the caller's ACK arrived.
        await placed.wait_established(timeout=10)
        await received.wait_established(timeout=10)
        return placed, received

    return connect


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


async def audio_arrives(call) -> bool:
    """True when decoded audio keeps arriving on the call."""
    with contextlib.suppress(AudioRestarted):  # the call renegotiates once on its own
        call.audio.read(1 << 20)
    call.audio.read(1 << 20)  # drain what has queued so far
    await asyncio.sleep(0.4)
    return len(call.audio.read(1 << 20)) > 0


async def unhurried(change) -> None:
    """Hold or resume, waiting out a renegotiation that is still in flight."""
    for _ in range(40):
        try:
            return await change()
        except BaresipError as error:
            if "in flight" not in str(error):
                raise
            await asyncio.sleep(0.05)
    await change()


async def until(condition, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not met in time"
        await asyncio.sleep(0.05)


@pytest.mark.parametrize("form", FORMS)
async def test_the_call_completes_and_the_callers_hangup_arrives(routed, form):
    placed, received = await routed(form)
    assert await audio_arrives(placed) and await audio_arrives(received)

    gone = closed(received)
    await placed.hangup()
    await asyncio.wait_for(gone, 2)


@pytest.mark.parametrize("form", FORMS)
async def test_the_answering_sides_hangup_arrives(routed, form):
    placed, received = await routed(form)

    gone = closed(placed)
    await received.hangup()
    await asyncio.wait_for(gone, 2)


@pytest.mark.parametrize("form", FORMS)
async def test_a_hold_from_the_answering_side_arrives(routed, form):
    placed, received = await routed(form)
    # Both sides renegotiate once on their own just after the call is
    # up; a hold sent into that would collide with it.
    await asyncio.sleep(0.6)

    await unhurried(received.hold)
    await until(lambda: placed.remote_on_hold)
    await unhurried(received.resume)
    await until(lambda: not placed.remote_on_hold)


@pytest.mark.parametrize("form", FORMS)
async def test_dtmf_sent_as_info_arrives(routed, form):
    placed, received = await routed(form, dtmf_mode="info")
    digits: list[str] = []
    placed.on_dtmf(lambda event: digits.append(event.digit))

    await received.send_dtmf("7")

    await until(lambda: digits == ["7"])


@pytest.mark.parametrize("form", FORMS)
async def test_a_transfer_request_arrives(routed, form):
    placed, received = await routed(form)
    requests: list = []
    placed.on_transfer_request(requests.append)

    # The caller's agent leaves the decision to the application, so the
    # request is delivered and nothing more happens.
    transfer = asyncio.create_task(received.transfer("sip:1005@127.0.0.1:15070"))

    await until(lambda: requests)
    assert "1005" in requests[0].target
    transfer.cancel()
