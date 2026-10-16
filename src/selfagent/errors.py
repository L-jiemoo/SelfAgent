"""Typed errors whose messages are meant to be shown to the user verbatim.

Anything raised as a SelfAgentError is a foreseeable condition: bad config,
wrong passphrase, unreachable endpoint. The CLI catches these and prints a
single readable line. Everything else is a bug and should surface as a
traceback so it gets reported rather than swallowed.
"""

from __future__ import annotations


class SelfAgentError(Exception):
    """Base class for errors that are safe and useful to show the user."""


class ConfigError(SelfAgentError):
    """config.json is missing, malformed, or internally inconsistent."""


class SecretsError(SelfAgentError):
    """secrets.enc is missing, corrupt, or the passphrase was wrong."""


class ProviderError(SelfAgentError):
    """A provider call failed in a way worth explaining (auth, quota, network)."""


class ToolsUnsupportedError(ProviderError):
    """The endpoint rejected the tools parameter outright.

    Distinct from ProviderError because it is recoverable: the caller retries
    the same request without tools rather than surfacing a failure. Support for
    function calling across OpenAI-compatible endpoints is inconsistent, and
    sending `tools` can break a request that would otherwise have worked.
    """


class ToolError(SelfAgentError):
    """A tool call could not be carried out (bad arguments, path outside the
    workspace, refused by the user). Reported back to the model as an error
    tool_result rather than aborting the turn."""
