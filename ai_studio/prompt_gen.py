"""Prompt generator — produces N image prompts across the 10-style catalog.

Ports the style rules from ``direct-response-ads/generating-ad-image-prompts/SKILL.md``
into a Python constant so we don't depend on the external skill file at
runtime. Each style has:
  - id / name / description (for UI)
  - cta_label + cta_color defaults (style-specific)
  - visual_cues (short snippets we splice into prompts)
  - template: a callable that emits a single finished prompt string

Every emitted prompt ends with the mandatory "Square format." sentinel.

If an ``AdInsights`` object is supplied, ``suggested_angles`` are rotated
through the prompts as primary text. When insights are missing or empty,
the offer's own ``headline`` / ``body`` / ``name`` are used.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SQUARE_SUFFIX = "Square format."
LANDSCAPE_SUFFIX = "16:9 landscape format, wide horizontal composition."


# --- Diversity micro-variations ---------------------------------------
# Injected into each prompt so the same offer doesn't collapse into
# identical renders every batch. Picked at random per prompt so two
# product_showcase images in one batch end up with different colour
# palettes, props, and camera angles.
_COLOR_PALETTES = [
    "cool blue and white tones",
    "warm beige and terracotta tones",
    "soft pastel pink and cream palette",
    "muted sage green and ivory palette",
    "neutral oat and grey tones",
    "dusty rose and charcoal accents",
    "pale blue and butter yellow palette",
    "rich espresso brown with warm honey highlights",
]

_CAMERA_ANGLES = [
    "shot at a slight three-quarter angle",
    "flat overhead angle",
    "eye-level frontal framing",
    "low angle looking slightly up",
    "dutch angle with subtle tilt",
    "side profile composition",
    "close-up with shallow depth of field",
    "wide environmental framing",
]

_SUBJECT_VARIATIONS = [
    "a middle-aged woman with shoulder-length brown hair",
    "a man in his late 50s with salt-and-pepper beard",
    "a woman in her early 60s with short silver hair and glasses",
    "a man in his mid 40s with a neatly trimmed beard",
    "a woman in her 50s with warm smile lines",
    "an older gentleman in his late 60s with kind eyes",
    "a woman in her late 40s with curly auburn hair",
    "a man in his early 50s wearing a simple sweater",
]

_LIGHTING_MOODS = [
    "morning window light with gentle shadows",
    "golden-hour warm side lighting",
    "overcast diffuse daylight",
    "bright noon light with crisp contrast",
    "soft indoor lamp glow",
    "cloudy afternoon light with muted tones",
    "directional studio light from the left",
    "ambient daylight bouncing off white walls",
]

_DECOR_PROPS = [
    "with a small potted succulent nearby",
    "with a folded linen napkin on the side",
    "next to a ceramic mug with steam rising",
    "with scattered fresh herbs and lemon slices",
    "beside a leather-bound notebook and fountain pen",
    "with a small glass of iced water condensing",
    "next to a woven rattan tray",
    "with a single white ceramic bowl alongside",
]


def _pick_variation(rng: random.Random, pool: List[str]) -> str:
    return pool[rng.randrange(len(pool))]


def _suffix_for_aspect(aspect: str) -> str:
    """Return the prompt suffix appropriate for the target aspect ratio.

    Only ``1:1`` (square) and ``16:9`` (landscape) are supported today —
    anything else falls back to square so we never emit an ambiguous prompt.
    """
    norm = (aspect or "").strip().lower().replace(" ", "")
    if norm in ("16:9", "landscape", "169"):
        return LANDSCAPE_SUFFIX
    return SQUARE_SUFFIX


def _retune_aspect(prompt: str, aspect: str) -> str:
    """Swap any trailing aspect sentinel (square/landscape) with the one that
    matches ``aspect``. Idempotent — safe to call on already-tuned prompts.
    """
    suffix = _suffix_for_aspect(aspect)
    text = prompt.rstrip()
    for sentinel in (SQUARE_SUFFIX, LANDSCAPE_SUFFIX):
        if text.endswith(sentinel):
            text = text[: -len(sentinel)].rstrip().rstrip(".")
            break
    if not text.endswith("."):
        text = text + "."
    return text + " " + suffix


@dataclass
class StyleDefinition:
    id: str
    name: str
    description: str
    visual_cues: List[str]
    default_cta_label: str
    default_cta_color: str
    emotional_fit: List[str]  # which emotional_triggers this style pairs best with
    template: Callable[[Dict[str, Any]], str] = field(repr=False)


def _pick_angle(
    insights: Optional[Dict[str, Any]],
    offer: Dict[str, Any],
    *,
    index: int,
) -> str:
    """Rotate through suggested angles, falling back to the offer headline."""
    angles: List[str] = []
    if insights:
        raw = insights.get("suggested_angles") or []
        if isinstance(raw, list):
            angles = [str(x).strip() for x in raw if str(x).strip()]
    if not angles:
        for key in ("headline", "name"):
            v = (offer.get(key) or "").strip()
            if v:
                angles = [v]
                break
    if not angles:
        angles = ["A simple, proven way to address the problem"]
    return angles[index % len(angles)]


def _pick_hook(insights: Optional[Dict[str, Any]], index: int) -> str:
    if not insights:
        return ""
    hooks = insights.get("top_hooks") or []
    if not isinstance(hooks, list) or not hooks:
        return ""
    return str(hooks[index % len(hooks)]).strip()


def _pick_mechanism(insights: Optional[Dict[str, Any]], index: int) -> str:
    if not insights:
        return ""
    mechs = insights.get("mechanisms") or []
    if not isinstance(mechs, list) or not mechs:
        return ""
    return str(mechs[index % len(mechs)]).strip()


# --- Style templates ---------------------------------------------------

def _tpl_product_showcase(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    brand = ctx.get("brand_name") or "the brand"
    cta = ctx.get("cta_label", "Learn More")
    return (
        f"A top-down editorial photograph on a clean light gray marble countertop showing the core "
        f"product or ingredient arranged alongside two complementary objects (a small bottle, a jar, or "
        f"a folded white towel) that hint at the concept of {brand}. A small handwritten note on torn "
        f"paper nearby reads a relatable line about the angle. Bold dark sans-serif text at the top of "
        f"the image reads \"{angle}\". A red rectangular button at the bottom center reads \"{cta}\" "
        f"in white text. Clean editorial product photography with soft natural light from the left. "
        f"{SQUARE_SUFFIX}"
    )


def _tpl_organic_native(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A casual overhead photo of a kitchen or desk surface with a coffee mug, a phone, and simple "
        f"everyday items. A pink or yellow sticky note stuck to the surface has handwritten blue ink "
        f"text reading \"{angle}\". The scene looks unposed, like someone grabbed their phone and took "
        f"a quick photo. Warm indoor lighting, slightly cluttered but authentic. No logos, no text "
        f"overlays on the image itself. {SQUARE_SUFFIX}"
    )


def _tpl_medical_illustration(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    mech = ctx.get("mechanism") or "the core mechanism"
    cta = ctx.get("cta_label", "See The Evidence")
    return (
        f"A clean flat vector illustration on a light beige background showing a stylised human anatomy "
        f"diagram (head/brain, joints, or torso depending on context) with a highlighted area related "
        f"to {mech} marked in bright red with small radiating lines. A dotted arrow leads from the "
        f"highlighted region toward a second labelled structure. Dark navy sans-serif text at the top "
        f"reads \"{angle}\". A soft blue rounded button at the bottom reads \"{cta}\" in white text. "
        f"Medical illustration style with clean lines and muted anatomical colors. {SQUARE_SUFFIX}"
    )


def _tpl_fake_social_proof(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A graphic designed to look like a Facebook post on a light gray background. Bold black text "
        f"at the top reads \"{angle}\". Below is a small image of a relatable middle-aged person with "
        f"a thoughtful expression. Three user comments appear below with small circular profile photos: "
        f"\"Sarah M.\" writes a short supportive line with 412 likes, \"Tom R.\" writes a confirming "
        f"line with 287 likes, and \"Linda K.\" writes a surprised line with 198 likes. A \"Write a "
        f"comment...\" text field at the bottom with a blue \"SEE HERE\" button in the lower right "
        f"corner. Realistic social-media interface design. {SQUARE_SUFFIX}"
    )


def _tpl_bold_text_urgency(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"].upper()
    cta = ctx.get("cta_label", "WATCH NOW").upper()
    return (
        f"A solid black background filling the entire frame. Large bold yellow text centered vertically "
        f"reads \"WARNING:\" at the top in underlined all-caps, followed by \"{angle}\" in slightly "
        f"smaller bold yellow text. A large orange rectangular button with rounded corners at the "
        f"bottom center reads \"{cta}\" in bold black text. No imagery, only bold typography on pure "
        f"black. {SQUARE_SUFFIX}"
    )


def _tpl_ugc_selfie(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A selfie-style photo of a relatable person in their late 40s to mid 50s, wearing casual "
        f"clothing, holding their phone at a slight angle in what appears to be their living room or "
        f"kitchen. They are looking directly at the camera with a sincere, slightly concerned "
        f"expression. White text overlay with a small drop shadow at the bottom of the image reads "
        f"\"{angle}\". The photo has natural afternoon window light, slightly warm tones, and feels "
        f"like a real social-media post. {SQUARE_SUFFIX}"
    )


def _tpl_before_after_split(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A horizontally split image. The left half has a muted blue-gray tone showing a person in "
        f"their 50s experiencing the core problem in a dim environment, with small white text at the "
        f"top reading \"BEFORE.\" The right half has warm golden tones showing the same person "
        f"relaxed and smiling in a bright setting, with small white text at the top reading \"AFTER.\" "
        f"Below the split, a thin white line runs vertically and a single centred caption reads "
        f"\"{angle}\". Cinematic lifestyle photography with dramatic tonal contrast between the two "
        f"sides. {SQUARE_SUFFIX}"
    )


def _tpl_news_headline(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A screenshot-style graphic designed to look like a health news article webpage. A red "
        f"\"HEALTH ALERT\" banner at the top left. A large serif headline reads \"{angle}\". Below the "
        f"headline is a smaller gray sans-serif line reading \"Published by National Research Desk | "
        f"February 2026.\" A thumbnail image to the right shows a neutral supporting photograph. A "
        f"byline reads \"By Dr. R. Harmon, Independent Research.\" The background is clean white with "
        f"subtle gray horizontal rules. Realistic news-website layout. {SQUARE_SUFFIX}"
    )


def _tpl_text_message(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"An iPhone 15 screen mockup on a soft gray background showing a text message conversation. "
        f"The first blue bubble reads \"Have you heard about this?\" The gray reply reads \"No why?\" "
        f"The next blue bubble reads \"{angle}\". The gray reply reads \"Wait what??\" The final blue "
        f"bubble reads \"Watch this video. Seriously. It explains everything.\" and below it a link "
        f"preview box shows a thumbnail image with a short title line. Realistic iMessage interface "
        f"with standard iOS fonts and spacing. {SQUARE_SUFFIX}"
    )


def _tpl_listicle(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    cta = ctx.get("cta_label", "Learn How")
    return (
        f"A clean white infographic with a bold sans-serif headline centered at the top reading "
        f"\"{angle}\". Below are four numbered steps arranged vertically, each with a small round "
        f"icon to the left (ear/brain/calendar/warning-triangle or equivalent for the offer). Each "
        f"step has a short one-line caption. Below the steps is a bold red text line emphasising the "
        f"urgency of addressing the root cause, and a blue rounded button reads \"{cta}\" in white "
        f"text. Minimal flat-design infographic with clean lines. {SQUARE_SUFFIX}"
    )


# --- Native-first styles (added for better platform blend) -----------

def _tpl_news_feed_story(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    brand = ctx.get("brand_name") or "a local resident"
    return (
        f"A smartphone screen mock-up showing a NewsBreak-style local news feed article card. Top of "
        f"the card has a small thumbnail (candid photo of a middle-aged person in a kitchen or living "
        f"room) and a small serif headline that reads \"{angle}\". Below the headline a gray metadata "
        f"line reads \"Local News · 2h ago · 14.2k reads\". Further down a 2-line preview snippet "
        f"mentions {brand} in a matter-of-fact tone. Faint vertical scroll-bar on the right edge. "
        f"Background is off-white with subtle news-feed chrome (share icon, comment bubble, small "
        f"thumbs-up). All text sits with at least 8% padding from every edge. The whole composition "
        f"feels like a reader-submitted local story, NOT a sponsored ad. {SQUARE_SUFFIX}"
    )


def _tpl_reddit_style_post(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A Reddit-style post mock-up on an off-white background. Top row: small orange Reddit-style "
        f"upvote arrow, a subreddit tag reading \"r/AskOldPeople · Posted by u/quiet_garden_48 · 5h\", "
        f"and a three-dot menu icon. Bold dark headline below reads \"{angle}\". Two lines of body "
        f"text mention a personal, first-hand discovery in a humble, curious tone. Bottom bar shows "
        f"\"842 upvotes\", \"207 comments\", \"Share\", and \"Save\" controls in muted gray. No "
        f"logos, no CTA button — the image should feel like an organic post someone would screenshot "
        f"and send a friend. Text has at least 8% margin from every edge. {SQUARE_SUFFIX}"
    )


def _tpl_forum_thread(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A web-forum thread mock-up on a light gray background with a plain sans-serif font. "
        f"Top header bar reads \"HealthBoards › Wellness & Recovery › Thread\". Bold navy thread title "
        f"reads \"{angle}\". Below are three stacked replies: \"Margaret_54\" (avatar: small blue "
        f"circle, icon of flower) writes 2 lines about her own experience; \"RetiredRN_Pat\" (avatar: "
        f"green circle with stethoscope icon) adds a short confirming reply; \"GrampsMike\" (avatar: "
        f"gray circle) writes a one-line thank-you. Each reply shows a post-count badge and a tiny "
        f"timestamp. The visual should look like a real old-school patient-support forum, not a "
        f"modern glossy ad. All text at least 8% from every edge. {SQUARE_SUFFIX}"
    )


def _tpl_x_tweet_embed(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A Twitter/X post mock-up embedded on a soft light-gray card. Top row: round profile photo "
        f"of a middle-aged man with a neutral smile, display name \"Dr. Ray Holden\", handle "
        f"\"@drrayholden\", and a small blue verified checkmark. Post body text in black sans-serif "
        f"reads: \"{angle}\" on two lines, followed by a single thoughtful follow-up sentence and a "
        f"soft \"🧵\" thread indicator. Below the post a muted metric row shows \"2:14 PM · Feb 18, "
        f"2026\", \"1,284 Reposts\", \"612 Quotes\", \"9,203 Likes\", \"184 Bookmarks\". Under that a "
        f"reply, repost, like, and share icon row. White card background with a subtle 1 px border. "
        f"No CTA button. Text has at least 8% margin from every edge. {SQUARE_SUFFIX}"
    )


def _tpl_local_news_alert(ctx: Dict[str, Any]) -> str:
    angle = ctx["angle"]
    return (
        f"A mobile news-alert card mock-up designed to look exactly like a push notification from a "
        f"local news app. Top-left: small square app icon (stylised news building silhouette in "
        f"crimson). App name \"Local Desk\" in bold 12pt, timestamp \"now\" in 10pt gray to the "
        f"right. The alert title in 16pt semi-bold dark navy reads \"{angle}\". A smaller 13pt gray "
        f"sub-line reads \"Tap to read the full story from Local Desk Health.\". The card has a "
        f"subtle rounded-corner drop-shadow on a pale gray phone-home-screen background with a "
        f"faint clock widget blurred in the upper portion. No CTA button, no brand logo — it must "
        f"read as a genuine news alert, not an ad. All text sits at least 8% from every edge. "
        f"{SQUARE_SUFFIX}"
    )


STYLE_CATALOG: List[StyleDefinition] = [
    StyleDefinition(
        id="product_showcase",
        name="Product / Ingredient Showcase",
        description="Editorial top-down photo of the product or ingredient on a clean surface.",
        visual_cues=["marble countertop", "editorial flat lay", "soft natural light"],
        default_cta_label="Learn More",
        default_cta_color="#DC2626",
        emotional_fit=["relief/simplicity", "authority/trust"],
        template=_tpl_product_showcase,
    ),
    StyleDefinition(
        id="organic_native_photo",
        name="Organic Native Photo",
        description="Casual phone-shot scene with a handwritten sticky note.",
        visual_cues=["sticky note", "phone snapshot", "unposed scene"],
        default_cta_label="See How",
        default_cta_color="#10B981",
        emotional_fit=["relief/simplicity", "curiosity"],
        template=_tpl_organic_native,
    ),
    StyleDefinition(
        id="medical_illustration",
        name="Medical / Scientific Illustration",
        description="Flat vector anatomy diagram highlighting the mechanism.",
        visual_cues=["flat vector", "anatomical diagram", "dotted arrows"],
        default_cta_label="See The Evidence",
        default_cta_color="#2563EB",
        emotional_fit=["authority/trust", "curiosity"],
        template=_tpl_medical_illustration,
    ),
    StyleDefinition(
        id="fake_social_proof",
        name="Fake Social Proof",
        description="Mocked Facebook-style post with 3 testimonial comments.",
        visual_cues=["comment thread", "profile thumbnails", "like counts"],
        default_cta_label="See Here",
        default_cta_color="#2563EB",
        emotional_fit=["social proof", "curiosity"],
        template=_tpl_fake_social_proof,
    ),
    StyleDefinition(
        id="bold_text_urgency",
        name="Bold Text Urgency",
        description="Solid black background, oversize yellow warning type, orange CTA.",
        visual_cues=["black background", "oversize yellow type", "orange CTA"],
        default_cta_label="WATCH NOW",
        default_cta_color="#F97316",
        emotional_fit=["fear/urgency"],
        template=_tpl_bold_text_urgency,
    ),
    StyleDefinition(
        id="ugc_selfie",
        name="UGC Selfie",
        description="Relatable middle-aged person selfie with caption overlay.",
        visual_cues=["selfie angle", "natural light", "caption overlay"],
        default_cta_label="Watch Now",
        default_cta_color="#DC2626",
        emotional_fit=["relief/simplicity", "social proof"],
        template=_tpl_ugc_selfie,
    ),
    StyleDefinition(
        id="before_after_split",
        name="Before / After Split",
        description="Horizontally split muted-to-warm before/after comparison.",
        visual_cues=["split composition", "desaturated vs warm", "BEFORE/AFTER labels"],
        default_cta_label="See The Change",
        default_cta_color="#10B981",
        emotional_fit=["relief/simplicity", "transformation"],
        template=_tpl_before_after_split,
    ),
    StyleDefinition(
        id="news_headline",
        name="News Headline / Article",
        description="Fake health-news article screenshot with serif headline.",
        visual_cues=["news layout", "serif headline", "HEALTH ALERT banner"],
        default_cta_label="Read Now",
        default_cta_color="#DC2626",
        emotional_fit=["authority/trust", "fear/urgency"],
        template=_tpl_news_headline,
    ),
    StyleDefinition(
        id="text_message",
        name="Text Message / Phone Screen",
        description="iMessage-style screen mockup with a two-person conversation.",
        visual_cues=["iMessage UI", "blue/gray bubbles", "link preview"],
        default_cta_label="Watch Now",
        default_cta_color="#2563EB",
        emotional_fit=["curiosity", "social proof"],
        template=_tpl_text_message,
    ),
    StyleDefinition(
        id="listicle",
        name="Listicle / Numbered Steps",
        description="Clean numbered-steps infographic on white background.",
        visual_cues=["numbered steps", "small round icons", "flat infographic"],
        default_cta_label="Learn How",
        default_cta_color="#2563EB",
        emotional_fit=["curiosity", "authority/trust"],
        template=_tpl_listicle,
    ),
    # --- native-first styles ---------------------------------------------
    StyleDefinition(
        id="news_feed_story",
        name="News-Feed Story Card",
        description="NewsBreak-style local-news article card mock-up (organic, not sponsored).",
        visual_cues=["news-feed card", "local-news chrome", "reader-submitted feel"],
        default_cta_label="Read Story",
        default_cta_color="#1F2937",
        emotional_fit=["authority/trust", "curiosity", "social proof"],
        template=_tpl_news_feed_story,
    ),
    StyleDefinition(
        id="reddit_style_post",
        name="Reddit-Style Post",
        description="Organic r/AskOldPeople-style post mock-up, no CTA.",
        visual_cues=["Reddit chrome", "subreddit header", "upvote counter"],
        default_cta_label="Read Post",
        default_cta_color="#FF4500",
        emotional_fit=["social proof", "curiosity"],
        template=_tpl_reddit_style_post,
    ),
    StyleDefinition(
        id="forum_thread",
        name="Old-School Forum Thread",
        description="Patient-support forum thread with stacked testimonial replies.",
        visual_cues=["forum breadcrumbs", "avatar replies", "thread title"],
        default_cta_label="Read Thread",
        default_cta_color="#1E3A8A",
        emotional_fit=["social proof", "authority/trust"],
        template=_tpl_forum_thread,
    ),
    StyleDefinition(
        id="x_tweet_embed",
        name="X / Twitter Embed",
        description="Verified-expert tweet mock-up with engagement metrics, no CTA.",
        visual_cues=["verified check", "tweet metrics", "thread indicator"],
        default_cta_label="View Post",
        default_cta_color="#0F172A",
        emotional_fit=["authority/trust", "social proof"],
        template=_tpl_x_tweet_embed,
    ),
    StyleDefinition(
        id="local_news_alert",
        name="Local News Push Alert",
        description="Push-notification card that reads as a genuine local news alert.",
        visual_cues=["push notification", "app icon", "timestamp badge"],
        default_cta_label="Read Alert",
        default_cta_color="#DC2626",
        emotional_fit=["fear/urgency", "authority/trust"],
        template=_tpl_local_news_alert,
    ),
]


STYLE_BY_ID: Dict[str, StyleDefinition] = {s.id: s for s in STYLE_CATALOG}


def generate_prompts(
    offer: Dict[str, Any],
    insights: Optional[Dict[str, Any]] = None,
    *,
    count: int = 10,
    style_mix: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    aspect: str = "1:1",
) -> List[Dict[str, Any]]:
    """Return ``count`` prompts, one per style (cycling if count > 10).

    Parameters
    ----------
    offer : dict
        Saved offer (storage.list_offers() row). Provides brand, CTA, headline.
    insights : dict or None
        :class:`AdInsights` digest from analyzer. When present, ``top_hooks``
        and ``suggested_angles`` take priority over the offer's own copy.
    count : int
        How many prompts to emit. Defaults to 10 (one per catalog style).
    style_mix : sequence of str or None
        Explicit ordered list of style ids. When provided, overrides the
        default "rotate all 10 styles" behaviour. Unknown ids are skipped.
    seed : int or None
        Deterministic seed — useful for tests / reproducibility.
    """
    rng = random.Random(seed)

    chosen: List[StyleDefinition]
    if style_mix:
        chosen = [STYLE_BY_ID[s] for s in style_mix if s in STYLE_BY_ID]
        if not chosen:
            chosen = list(STYLE_CATALOG)
    else:
        chosen = list(STYLE_CATALOG)

    # When count > len(chosen), repeat through the list.
    picks: List[StyleDefinition] = []
    for i in range(count):
        picks.append(chosen[i % len(chosen)])

    # A fresh per-call RNG so subsequent Generate clicks produce DIFFERENT
    # variations even on the same offer/insights. Seeded RNG (if requested)
    # still yields reproducible output for tests.
    var_rng = random.Random(seed if seed is not None else None)

    out: List[Dict[str, Any]] = []
    offer_cta = (offer.get("cta") or "").strip()
    offer_brand = (offer.get("brand_name") or offer.get("name") or "").strip()
    for i, style in enumerate(picks):
        angle = _pick_angle(insights, offer, index=i)
        hook = _pick_hook(insights, index=i) or angle
        mechanism = _pick_mechanism(insights, index=i) or ""
        # Per-prompt diversity tokens — same style can produce visually
        # distinct renders across the batch.
        palette = _pick_variation(var_rng, _COLOR_PALETTES)
        camera = _pick_variation(var_rng, _CAMERA_ANGLES)
        subject = _pick_variation(var_rng, _SUBJECT_VARIATIONS)
        lighting = _pick_variation(var_rng, _LIGHTING_MOODS)
        prop = _pick_variation(var_rng, _DECOR_PROPS)
        variation_id = var_rng.randrange(10_000_000)
        ctx = {
            "angle": angle,
            "hook": hook,
            "mechanism": mechanism,
            "brand_name": offer_brand or "the offer",
            "cta_label": offer_cta or style.default_cta_label,
            "cta_color": style.default_cta_color,
        }
        base_prompt = style.template(ctx).strip()
        # Inject variation directives before the aspect suffix so the
        # provider actually sees them as style modifiers, not trailing noise.
        variation_block = (
            f" Visual variation: {palette}, {lighting}, {camera}. "
            f"Subject reference when a person is shown: {subject}. "
            f"Decor accent: {prop}. Creative seed v{variation_id}."
        )
        # Strip any existing aspect suffix, add variation, re-tune aspect.
        stripped = base_prompt
        for sentinel in (SQUARE_SUFFIX, LANDSCAPE_SUFFIX):
            if stripped.endswith(sentinel):
                stripped = stripped[: -len(sentinel)].rstrip()
                break
        prompt = _retune_aspect(stripped + variation_block, aspect)
        out.append(
            {
                "style_id": style.id,
                "style_name": style.name,
                "prompt": prompt,
                "cta_label": ctx["cta_label"],
                "cta_color": ctx["cta_color"],
                "angle": angle,
                "aspect": aspect,
                "variation_id": variation_id,
            }
        )
    # Mild shuffle on seed to avoid the UI always starting with "product_showcase".
    if seed is None:
        rng2 = random.Random()
        rng2.shuffle(out)
    return out


__all__ = [
    "STYLE_CATALOG",
    "STYLE_BY_ID",
    "StyleDefinition",
    "generate_prompts",
    "SQUARE_SUFFIX",
    "LANDSCAPE_SUFFIX",
]
