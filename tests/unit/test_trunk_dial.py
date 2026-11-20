#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Dialing from an account that does not register.

Such an account gives the stack nothing to take a call's address family
from when the target is a host name, and a media-NAT session needs one to
look its server up. These dials reach no network: the names end in
``.invalid``, so each call closes on its own once resolution fails. What
is under test is that the dial is accepted at all.
"""

import pytest

pytest.importorskip("baresip._native")

from baresip import Account, Config, Runtime, UserAgent

TARGET = "sip:+15551234567@trunk.invalid"


@pytest.fixture
async def runtime():
    rt = Runtime()
    await rt.start(Config(net_interface="127.0.0.1"))
    try:
        yield rt
    finally:
        await rt.close()


async def trunk_ua(runtime, user, *extra_params):
    return await UserAgent.create(
        runtime,
        Account(
            user=user,
            password="",
            domain="trunk.invalid",
            reg_interval=0,
            extra_params=extra_params,
        ),
    )


@pytest.mark.parametrize("medianat", ["stun", "ice"])
async def test_trunk_dial_to_a_host_name_with_media_nat(runtime, medianat):
    ua = await trunk_ua(runtime, medianat, f"medianat={medianat}", "stunserver=stun:stun.invalid")

    call = await ua.dial(TARGET)

    assert call.peer == TARGET


async def test_media_nat_dial_leaves_later_dials_alone(runtime):
    # The family chosen for the media-NAT dial is not left behind for an
    # account that has no media NAT.
    with_nat = await trunk_ua(runtime, "nat", "medianat=stun", "stunserver=stun:stun.invalid")
    plain = await trunk_ua(runtime, "plain")

    await with_nat.dial(TARGET)
    call = await plain.dial(TARGET)

    assert call.peer == TARGET
