#!/usr/bin/env python3
"""Social media posting agent for Facebook, Instagram, and LinkedIn.

Usage example:
  python3 scripts/social_post_agent.py \
    --platforms facebook,linkedin \
    --message "EQ AI Tuner v0.1.1 is live" \
    --link "https://github.com/ben2079/eq-ai-tuner" \
    --no-dry-run

Environment variables:
  FACEBOOK_PAGE_ID
  FACEBOOK_PAGE_ACCESS_TOKEN
  INSTAGRAM_BUSINESS_ACCOUNT_ID
  INSTAGRAM_ACCESS_TOKEN
  LINKEDIN_AUTHOR_URN            (e.g. urn:li:person:xxxxxxxx)
  LINKEDIN_ACCESS_TOKEN
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

import requests

FB_API_VERSION = "v21.0"
IG_API_VERSION = "v21.0"


class PostingError(RuntimeError):
    pass


class SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PostingError(f"Missing environment variable: {name}")
    return value


def post_facebook(message: str, link: Optional[str], dry_run: bool) -> str:
    page_id = os.environ.get("FACEBOOK_PAGE_ID", "<FACEBOOK_PAGE_ID>") if dry_run else require_env("FACEBOOK_PAGE_ID")
    token = (
        os.environ.get("FACEBOOK_PAGE_ACCESS_TOKEN", "<FACEBOOK_PAGE_ACCESS_TOKEN>")
        if dry_run
        else require_env("FACEBOOK_PAGE_ACCESS_TOKEN")
    )

    payload = {"message": message, "access_token": token}
    if link:
        payload["link"] = link

    url = f"https://graph.facebook.com/{FB_API_VERSION}/{page_id}/feed"

    if dry_run:
        return f"[dry-run] facebook -> {url} payload_keys={sorted(payload.keys())}"

    res = requests.post(url, data=payload, timeout=30)
    if not res.ok:
        raise PostingError(f"Facebook post failed: {res.status_code} {res.text}")

    post_id = res.json().get("id", "unknown")
    return f"facebook posted: {post_id}"


def post_instagram(caption: str, image_url: str, dry_run: bool) -> str:
    if not image_url:
        raise PostingError("Instagram requires --image-url")

    business_id = (
        os.environ.get("INSTAGRAM_BUSINESS_ACCOUNT_ID", "<INSTAGRAM_BUSINESS_ACCOUNT_ID>")
        if dry_run
        else require_env("INSTAGRAM_BUSINESS_ACCOUNT_ID")
    )
    token = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "<INSTAGRAM_ACCESS_TOKEN>") if dry_run else require_env("INSTAGRAM_ACCESS_TOKEN")

    create_url = f"https://graph.facebook.com/{IG_API_VERSION}/{business_id}/media"
    publish_url = f"https://graph.facebook.com/{IG_API_VERSION}/{business_id}/media_publish"

    create_payload = {
        "image_url": image_url,
        "caption": caption,
        "access_token": token,
    }

    if dry_run:
        return (
            f"[dry-run] instagram create={create_url} publish={publish_url} "
            f"payload_keys={sorted(create_payload.keys())}"
        )

    create_res = requests.post(create_url, data=create_payload, timeout=30)
    if not create_res.ok:
        raise PostingError(f"Instagram create media failed: {create_res.status_code} {create_res.text}")

    creation_id = create_res.json().get("id")
    if not creation_id:
        raise PostingError("Instagram create media returned no id")

    # Short wait to avoid immediate publish race on some accounts.
    time.sleep(2)

    publish_payload = {"creation_id": creation_id, "access_token": token}
    publish_res = requests.post(publish_url, data=publish_payload, timeout=30)
    if not publish_res.ok:
        raise PostingError(f"Instagram publish failed: {publish_res.status_code} {publish_res.text}")

    media_id = publish_res.json().get("id", "unknown")
    return f"instagram posted: {media_id}"


def post_linkedin(message: str, link: Optional[str], dry_run: bool) -> str:
    author = os.environ.get("LINKEDIN_AUTHOR_URN", "urn:li:person:<LINKEDIN_AUTHOR_URN>") if dry_run else require_env("LINKEDIN_AUTHOR_URN")
    token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "<LINKEDIN_ACCESS_TOKEN>") if dry_run else require_env("LINKEDIN_ACCESS_TOKEN")

    commentary = message if not link else f"{message}\n\n{link}"

    payload = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": commentary},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    url = "https://api.linkedin.com/v2/ugcPosts"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }

    if dry_run:
        return f"[dry-run] linkedin -> {url} payload_keys={sorted(payload.keys())}"

    res = requests.post(url, json=payload, headers=headers, timeout=30)
    if not res.ok:
        raise PostingError(f"LinkedIn post failed: {res.status_code} {res.text}")

    post_urn = res.headers.get("x-restli-id", "unknown")
    return f"linkedin posted: {post_urn}"


def parse_platforms(raw: str) -> List[str]:
    platforms = [p.strip().lower() for p in raw.split(",") if p.strip()]
    valid = {"facebook", "instagram", "linkedin"}
    invalid = [p for p in platforms if p not in valid]
    if invalid:
        raise PostingError(f"Invalid platforms: {', '.join(invalid)}")
    if not platforms:
        raise PostingError("No platforms specified")
    return platforms


def parse_template_vars(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise PostingError(f"Invalid --template-var '{item}', expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise PostingError(f"Invalid --template-var '{item}', empty key")
        out[k] = v
    return out


def build_message(
    message: str,
    template_file: str,
    template_vars: Dict[str, str],
    link: Optional[str],
) -> str:
    if not template_file:
        return message

    if not os.path.exists(template_file):
        raise PostingError(f"Template file not found: {template_file}")

    with open(template_file, "r", encoding="utf-8") as f:
        template = f.read()

    merged = dict(template_vars)
    if link and "link" not in merged:
        merged["link"] = link

    rendered = template.format_map(SafeFormatDict(merged)).strip()
    if not rendered:
        raise PostingError("Rendered template is empty")
    return rendered


def post_to_platforms(
    platforms: Iterable[str],
    message: str,
    link: Optional[str],
    image_url: Optional[str],
    dry_run: bool,
) -> List[str]:
    results: List[str] = []
    for platform in platforms:
        if platform == "facebook":
            results.append(post_facebook(message, link, dry_run))
        elif platform == "instagram":
            caption = message if not link else f"{message}\n\n{link}"
            results.append(post_instagram(caption, image_url or "", dry_run))
        elif platform == "linkedin":
            results.append(post_linkedin(message, link, dry_run))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post to Facebook/Instagram/LinkedIn via official APIs")
    parser.add_argument(
        "--platforms",
        required=True,
        help="Comma-separated: facebook,instagram,linkedin",
    )
    parser.add_argument("--message", required=True, help="Post text")
    parser.add_argument("--template-file", default="", help="Optional text template file for message body")
    parser.add_argument(
        "--template-var",
        action="append",
        default=[],
        help="Template variable key=value (repeatable)",
    )
    parser.add_argument("--link", default="", help="Optional link appended to message")
    parser.add_argument("--image-url", default="", help="Required for Instagram image post")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview API calls only (default true)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually execute API calls",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        platforms = parse_platforms(args.platforms)
        vars_map = parse_template_vars(args.template_var)
        final_message = build_message(
            message=args.message,
            template_file=args.template_file,
            template_vars=vars_map,
            link=args.link or None,
        )
        results = post_to_platforms(
            platforms=platforms,
            message=final_message,
            link=args.link or None,
            image_url=args.image_url or None,
            dry_run=args.dry_run,
        )
        for line in results:
            print(line)
        return 0
    except PostingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
