"""Tests for the CLI setup."""

import logging

from sup7.cli import setup_logging


def test_httpx_request_urls_are_not_logged():
    # the Workers AI URL carries the Cloudflare account id
    setup_logging(verbose=False)
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
