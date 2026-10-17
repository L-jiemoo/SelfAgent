import io

from selfagent import banner
from selfagent.render import Renderer, _lerp_rgb


class Tty(io.StringIO):
    """Reports a real terminal. `encoding` must be a class attribute --
    io.StringIO's own is a read-only descriptor."""

    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class GbkTty(Tty):
    """A Chinese console without chcp 65001: no U+2580 in its codepage."""

    encoding = "gbk"


class Spec:
    def __init__(self, models):
        self.models = models


PROVIDERS = {
    "deepseek": Spec(["deepseek-v4-flash", "deepseek-v4-pro"]),
    "qwen": Spec(["qwen-max"]),
    "custom": Spec([]),
}


def lines(mounted=("deepseek",)):
    return banner.info_lines("1.0.0", "deepseek/deepseek-v4-flash", r"D:\x", PROVIDERS, set(mounted))


def splash(out, info, color: bool) -> str:
    Renderer(color=color, out=out, err=io.StringIO()).splash(
        banner.LOGO, info, top=banner.COLOR_TOP, bottom=banner.COLOR_BOTTOM
    )
    return out.getvalue()


def text_of(info):
    return [t for t, _ in info]


# ---- logo ------------------------------------------------------------


def test_logo_is_rectangular() -> None:
    # The art is transcribed from a rasteriser; an off-by-one on any row shows
    # up as a ragged edge, so pin the width.
    widths = {len(line) for line in banner.LOGO}
    assert len(widths) == 1
    assert widths.pop() == 44
    assert len(banner.LOGO) == 6


def test_logo_fits_a_standard_terminal() -> None:
    assert len(banner.LOGO[0]) <= 80


def test_every_line_fits_a_standard_terminal() -> None:
    too_long = [t for t in text_of(lines()) if len(t) > 79]
    assert too_long == [], too_long


def test_gradient_endpoints() -> None:
    assert _lerp_rgb(banner.COLOR_TOP, banner.COLOR_BOTTOM, 0.0) == banner.COLOR_TOP
    assert _lerp_rgb(banner.COLOR_TOP, banner.COLOR_BOTTOM, 1.0) == banner.COLOR_BOTTOM


# ---- content ---------------------------------------------------------


def test_shows_version_and_what_the_tool_is() -> None:
    body = "\n".join(text_of(lines()))
    assert "SelfAgent v1.0.0" in body
    assert "便携式多模型" in body
    assert banner.TAGLINE in body


def test_a_configured_agent_name_replaces_the_product_name() -> None:
    info = banner.info_lines(
        "1.0.0", "deepseek/deepseek-v4-flash", r"D:\x", PROVIDERS, set(), name="小助手"
    )
    body = "\n".join(text_of(info))
    assert "小助手 v1.0.0" in body
    assert "SelfAgent" not in body


def test_splash_uses_the_configured_theme() -> None:
    renderer = Renderer(color=True, out=Tty(), err=io.StringIO())
    renderer.set_theme((255, 0, 0), (0, 0, 255), (255, 0, 0))
    renderer.splash(banner.LOGO, [("标题", "bold")])

    out = renderer.out.getvalue()
    assert "38;2;255;0;0" in out
    assert "38;2;0;0;255" in out


def test_the_degraded_logo_uses_the_agent_name() -> None:
    renderer = Renderer(color=True, out=GbkTty(), err=io.StringIO())
    renderer.set_agent_name("小助手")
    renderer.splash(banner.LOGO, [("标题", "bold")])

    out = renderer.out.getvalue()
    assert "小助手" in out
    assert "▀" not in out


def test_lists_every_provider_with_its_models() -> None:
    body = "\n".join(text_of(lines()))
    for name, spec in PROVIDERS.items():
        assert name in body
        for model in spec.models:
            assert model in body


def test_provider_with_no_models_gets_a_placeholder() -> None:
    body = "\n".join(text_of(lines()))
    assert banner.NO_MODELS in body


def test_mounted_providers_are_marked() -> None:
    body = "\n".join(text_of(lines(mounted=["deepseek", "qwen"])))
    assert f"  * {'deepseek':<9}" in body
    assert f"  * {'qwen':<9}" in body


def test_unmounted_providers_are_shown_but_dimmed() -> None:
    info = lines(mounted=["deepseek"])
    style_of = {text: style for text, style in info}

    mounted_line = f"  * {'deepseek':<9} deepseek-v4-flash, deepseek-v4-pro"
    assert style_of[mounted_line] == ""

    unmounted_line = next(t for t, _ in info if t.strip().startswith("custom"))
    assert unmounted_line.startswith(f"    {'custom':<9}")
    assert style_of[unmounted_line] == "dim"


def test_shows_current_model_and_directory() -> None:
    body = "\n".join(text_of(lines()))
    assert "deepseek/deepseek-v4-flash" in body
    assert r"D:\x" in body


def test_shows_the_command_hints() -> None:
    assert "/help" in "\n".join(text_of(lines()))


# ---- rendering -------------------------------------------------------


def test_plain_splash_has_no_escape_codes() -> None:
    out = splash(io.StringIO(), lines(), color=False)
    assert "\x1b" not in out
    assert banner.LOGO[0].strip() in out


def test_colour_splash_uses_a_truecolor_gradient() -> None:
    out = splash(Tty(), lines(), color=True)
    assert "38;2;34;211;238" in out  # cyan end
    assert "38;2;168;85;247" in out  # violet end


def test_colour_splash_still_shows_info() -> None:
    out = splash(Tty(), lines(), color=True)
    assert "SelfAgent v1.0.0" in out
    assert "/help" in out


def test_degrades_when_console_cannot_encode_the_blocks() -> None:
    # U+2580 is absent from GBK, so a cp936 console without chcp 65001 cannot
    # print the logo. Startup must not be the thing that crashes.
    out = splash(GbkTty(), lines(), color=True)
    assert "SelfAgent v1.0.0" in out
    assert "▀" not in out


def test_all_splash_text_survives_a_gbk_console() -> None:
    # The fallback only swaps the logo; the info lines still have to encode.
    body = "\n".join(text_of(lines()) + [banner.TAGLINE, banner.HINTS])
    body.encode("gbk")
