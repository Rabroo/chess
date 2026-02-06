"""Social media scraper module."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from ..utils.downloader import Downloader, DownloadError
from ..utils.hashing import HashManager
from ..utils.validator import ValidationError
from .base import ScrapedItem, ScraperModule, ScraperRegistry


@ScraperRegistry.register("social")
class SocialScraper(ScraperModule):
    """Scraper for collecting social media posts and metadata."""

    MODULE_NAME = "social"
    ALLOWED_CONTENT_TYPES = {"application/json", "text/plain"}

    # Supported platforms
    SUPPORTED_PLATFORMS = {"twitter", "mastodon", "bluesky", "reddit", "youtube", "rss"}

    # Chrome User-Agent for scraping
    CHROME_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downloader = Downloader(
            timeout=self.config.network.timeout,
            retries=self.config.network.retries,
            retry_backoff=self.config.network.retry_backoff,
            user_agent=self.config.network.user_agent,
        )
        self._credentials_loaded = False
        self._api_keys: Dict[str, str] = {}

    def _load_credentials(self) -> None:
        """Load credentials from config or environment."""
        if self._credentials_loaded:
            return

        creds = self.config.credentials

        # Twitter credentials
        if creds.twitter_api_key:
            self._api_keys["twitter_api_key"] = creds.twitter_api_key
        if creds.twitter_api_secret:
            self._api_keys["twitter_api_secret"] = creds.twitter_api_secret
        if creds.twitter_access_token:
            self._api_keys["twitter_access_token"] = creds.twitter_access_token
        if creds.twitter_access_secret:
            self._api_keys["twitter_access_secret"] = creds.twitter_access_secret

        self._credentials_loaded = True

    def _has_credentials(self, platform: str) -> bool:
        """Check if credentials are available for a platform."""
        self._load_credentials()

        if platform == "twitter":
            return bool(
                self._api_keys.get("twitter_api_key") and
                self._api_keys.get("twitter_api_secret")
            )

        return False

    def _detect_platform(self, input_value: str) -> Optional[str]:
        """Detect the social media platform from input."""
        if "twitter.com" in input_value or "x.com" in input_value:
            return "twitter"
        elif "mastodon" in input_value:
            return "mastodon"
        elif "bsky" in input_value or "bluesky" in input_value:
            return "bluesky"
        elif "reddit.com" in input_value or input_value.startswith("r/"):
            return "reddit"
        elif "youtube.com" in input_value or "youtu.be" in input_value:
            return "youtube"
        elif input_value.endswith(".rss") or input_value.endswith("/feed") or "/rss" in input_value:
            return "rss"
        return None

    def validate_input(self, input_value: str) -> bool:
        """
        Validate input - can be URL, hashtag, user handle, or search term.

        Args:
            input_value: Post URL, hashtag (#tag), handle (@user), or search term.

        Returns:
            True if valid.

        Raises:
            ValidationError: If invalid.
        """
        # URL
        if input_value.startswith("http"):
            self.validator.validate_url(input_value)
            platform = self._detect_platform(input_value)
            if platform and not self._has_credentials(platform):
                self.logger.warning(
                    f"No credentials for {platform} - some features may be limited",
                    module=self.MODULE_NAME,
                )
            return True

        # Hashtag
        if input_value.startswith("#"):
            if len(input_value) < 2:
                raise ValidationError("Hashtag must have at least one character")
            return True

        # User handle
        if input_value.startswith("@"):
            if len(input_value) < 2:
                raise ValidationError("Handle must have at least one character")
            return True

        # File path with list of inputs
        if input_value.startswith("file:"):
            file_path = input_value[5:]
            if not Path(file_path).exists():
                raise ValidationError(f"File not found: {file_path}")
            return True

        # Search term - allow anything non-empty
        if len(input_value.strip()) > 0:
            return True

        raise ValidationError("Input cannot be empty")

    def _parse_post_url(self, url: str) -> Dict[str, Any]:
        """Parse a social media post URL to extract identifiers."""
        parsed = urlparse(url)
        result = {"url": url, "platform": self._detect_platform(url)}

        # Twitter/X URL patterns
        if result["platform"] == "twitter":
            # https://twitter.com/user/status/123456789
            match = re.search(r"/status/(\d+)", parsed.path)
            if match:
                result["post_id"] = match.group(1)
            # Extract username
            path_parts = parsed.path.strip("/").split("/")
            if path_parts:
                result["username"] = path_parts[0]

        return result

    def _fetch_reddit(self, subreddit: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch posts from a subreddit using Reddit's JSON API.

        Args:
            subreddit: Subreddit name (without r/ prefix).
            limit: Maximum posts to fetch.

        Yields:
            ScrapedItem for each post.
        """
        import requests

        self.logger.info(
            f"Fetching posts from r/{subreddit}",
            module=self.MODULE_NAME,
            action="FETCH_SUBREDDIT",
        )

        try:
            # Reddit JSON API - just append .json to the URL
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

            headers = {
                "User-Agent": self.CHROME_USER_AGENT,
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            data = response.json()
            posts = data.get("data", {}).get("children", [])

            self.logger.info(
                f"Found {len(posts)} posts in r/{subreddit}",
                module=self.MODULE_NAME,
                action="FETCH_COMPLETE",
            )

            for post_item in posts[:limit]:
                post = post_item.get("data", {})

                post_data = {
                    "id": post.get("id", ""),
                    "platform": "reddit",
                    "subreddit": subreddit,
                    "source_url": f"https://reddit.com{post.get('permalink', '')}",
                    "title": post.get("title", ""),
                    "content": post.get("selftext", ""),
                    "author": {
                        "username": post.get("author", ""),
                    },
                    "created_at": post.get("created_utc"),
                    "metrics": {
                        "score": post.get("score", 0),
                        "upvote_ratio": post.get("upvote_ratio", 0),
                        "num_comments": post.get("num_comments", 0),
                    },
                    "is_video": post.get("is_video", False),
                    "thumbnail": post.get("thumbnail", ""),
                    "url": post.get("url", ""),  # Link posts
                    "scraped_at": datetime.now().isoformat(),
                }

                json_content = json.dumps(post_data, indent=2).encode("utf-8")

                yield ScrapedItem(
                    content=json_content,
                    identifier=f"reddit_{subreddit}_{post_data['id']}.json",
                    metadata=post_data,
                    content_type="application/json",
                    source_url=post_data["source_url"],
                )

        except Exception as e:
            self.logger.error(
                f"Reddit fetch failed: {e}",
                module=self.MODULE_NAME,
            )

    def _fetch_youtube(self, video_id: str) -> Iterator[ScrapedItem]:
        """
        Fetch YouTube video metadata using oembed API.

        Args:
            video_id: YouTube video ID.

        Yields:
            ScrapedItem with video metadata.
        """
        import requests

        self.logger.info(
            f"Fetching YouTube video {video_id}",
            module=self.MODULE_NAME,
            action="FETCH_VIDEO",
        )

        try:
            # Use YouTube's oembed API (no auth needed)
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

            response = requests.get(oembed_url, timeout=15)
            response.raise_for_status()

            oembed_data = response.json()

            post_data = {
                "id": video_id,
                "platform": "youtube",
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": oembed_data.get("title", ""),
                "author": {
                    "name": oembed_data.get("author_name", ""),
                    "url": oembed_data.get("author_url", ""),
                },
                "thumbnail": oembed_data.get("thumbnail_url", ""),
                "thumbnail_width": oembed_data.get("thumbnail_width"),
                "thumbnail_height": oembed_data.get("thumbnail_height"),
                "scraped_at": datetime.now().isoformat(),
            }

            json_content = json.dumps(post_data, indent=2).encode("utf-8")

            yield ScrapedItem(
                content=json_content,
                identifier=f"youtube_{video_id}.json",
                metadata=post_data,
                content_type="application/json",
                source_url=post_data["source_url"],
            )

        except Exception as e:
            self.logger.error(
                f"YouTube fetch failed: {e}",
                module=self.MODULE_NAME,
            )

    def _fetch_rss(self, feed_url: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch articles from RSS/Atom feed.

        Args:
            feed_url: URL of the RSS/Atom feed.
            limit: Maximum items to fetch.

        Yields:
            ScrapedItem for each article.
        """
        import requests
        import xml.etree.ElementTree as ET

        self.logger.info(
            f"Fetching RSS feed: {feed_url}",
            module=self.MODULE_NAME,
            action="FETCH_RSS",
        )

        try:
            headers = {
                "User-Agent": self.CHROME_USER_AGENT,
            }

            response = requests.get(feed_url, headers=headers, timeout=15)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)

            items_yielded = 0

            # Handle RSS 2.0 format
            for item in root.findall(".//item"):
                if items_yielded >= limit:
                    break

                title = item.find("title")
                link = item.find("link")
                description = item.find("description")
                pub_date = item.find("pubDate")
                guid = item.find("guid")

                post_data = {
                    "id": guid.text if guid is not None else (link.text if link is not None else f"rss_{items_yielded}"),
                    "platform": "rss",
                    "source_url": link.text if link is not None else feed_url,
                    "feed_url": feed_url,
                    "title": title.text if title is not None else "",
                    "content": description.text if description is not None else "",
                    "published_at": pub_date.text if pub_date is not None else None,
                    "scraped_at": datetime.now().isoformat(),
                }

                json_content = json.dumps(post_data, indent=2).encode("utf-8")

                yield ScrapedItem(
                    content=json_content,
                    identifier=f"rss_{items_yielded}.json",
                    metadata=post_data,
                    content_type="application/json",
                    source_url=post_data["source_url"],
                )
                items_yielded += 1

            # Handle Atom format
            # Atom uses namespaces
            atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall(".//atom:entry", atom_ns):
                if items_yielded >= limit:
                    break

                title = entry.find("atom:title", atom_ns)
                link = entry.find("atom:link", atom_ns)
                summary = entry.find("atom:summary", atom_ns)
                content = entry.find("atom:content", atom_ns)
                published = entry.find("atom:published", atom_ns)
                entry_id = entry.find("atom:id", atom_ns)

                link_href = link.get("href") if link is not None else None

                post_data = {
                    "id": entry_id.text if entry_id is not None else f"atom_{items_yielded}",
                    "platform": "rss",
                    "source_url": link_href or feed_url,
                    "feed_url": feed_url,
                    "title": title.text if title is not None else "",
                    "content": (content.text if content is not None else
                               (summary.text if summary is not None else "")),
                    "published_at": published.text if published is not None else None,
                    "scraped_at": datetime.now().isoformat(),
                }

                json_content = json.dumps(post_data, indent=2).encode("utf-8")

                yield ScrapedItem(
                    content=json_content,
                    identifier=f"rss_{items_yielded}.json",
                    metadata=post_data,
                    content_type="application/json",
                    source_url=post_data["source_url"],
                )
                items_yielded += 1

            self.logger.info(
                f"Fetched {items_yielded} items from RSS feed",
                module=self.MODULE_NAME,
                action="FETCH_COMPLETE",
            )

        except Exception as e:
            self.logger.error(
                f"RSS fetch failed: {e}",
                module=self.MODULE_NAME,
            )

    def _create_placeholder_post(
        self,
        post_id: str,
        platform: str,
        source_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a placeholder post structure."""
        return {
            "id": post_id,
            "platform": platform,
            "source_url": source_url,
            "content": "",  # Would be populated by API
            "author": {},
            "created_at": None,
            "metrics": {
                "likes": 0,
                "reposts": 0,
                "replies": 0,
            },
            "media": [],
            "scraped_at": datetime.now().isoformat(),
        }

    def fetch(self, input_value: str, limit: int) -> Iterator[ScrapedItem]:
        """
        Fetch social media posts.

        Args:
            input_value: URL, hashtag, handle, or search term.
            limit: Maximum posts to fetch.

        Yields:
            ScrapedItem for each post.
        """
        items_yielded = 0

        # URL handling - dispatch to appropriate handler
        if input_value.startswith("http"):
            try:
                self.validator.validate_url(input_value)
                platform = self._detect_platform(input_value)

                # YouTube video URL
                if platform == "youtube":
                    # Extract video ID from URL
                    video_id = None
                    if "youtube.com/watch" in input_value:
                        match = re.search(r'v=([a-zA-Z0-9_-]{11})', input_value)
                        if match:
                            video_id = match.group(1)
                    elif "youtu.be/" in input_value:
                        match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', input_value)
                        if match:
                            video_id = match.group(1)

                    if video_id:
                        for item in self._fetch_youtube(video_id):
                            yield item
                            items_yielded += 1
                    else:
                        self.logger.warning(
                            f"Could not extract video ID from {input_value}",
                            module=self.MODULE_NAME,
                        )
                    return

                # Reddit URL
                if platform == "reddit":
                    # Extract subreddit from URL
                    match = re.search(r'reddit\.com/r/([^/]+)', input_value)
                    if match:
                        subreddit = match.group(1)
                        for item in self._fetch_reddit(subreddit, limit):
                            yield item
                            items_yielded += 1
                    else:
                        self.logger.warning(
                            f"Could not extract subreddit from {input_value}",
                            module=self.MODULE_NAME,
                        )
                    return

                # RSS/Atom feed URL
                if platform == "rss":
                    for item in self._fetch_rss(input_value, limit):
                        yield item
                        items_yielded += 1
                    return

                # Other platforms (Twitter, etc.) - use existing logic
                parsed = self._parse_post_url(input_value)

                if not parsed.get("post_id"):
                    self.logger.warning(
                        f"Could not extract post ID from {input_value}",
                        module=self.MODULE_NAME,
                    )
                    return

                post_id = parsed["post_id"]

                # In a real implementation, this would call the platform's API
                # For now, create a placeholder structure
                post_data = self._create_placeholder_post(
                    post_id, platform or "unknown", input_value
                )

                # If we have credentials, we could fetch full data
                if platform and self._has_credentials(platform):
                    self.logger.info(
                        f"Would fetch full post data for {post_id} from {platform}",
                        module=self.MODULE_NAME,
                    )

                json_content = json.dumps(post_data, indent=2).encode("utf-8")

                yield ScrapedItem(
                    content=json_content,
                    identifier=f"{platform or 'unknown'}_{post_id}.json",
                    metadata=post_data,
                    content_type="application/json",
                    source_url=input_value,
                )
                items_yielded += 1

            except ValidationError as e:
                self.logger.error(f"Invalid URL: {e}", module=self.MODULE_NAME)

        # Hashtag search
        elif input_value.startswith("#"):
            hashtag = input_value[1:]
            self.logger.info(
                f"Would search for hashtag #{hashtag} (requires API credentials)",
                module=self.MODULE_NAME,
            )

            # Placeholder - would yield posts from API search
            for i in range(min(limit, 1)):  # Just one placeholder
                post_data = self._create_placeholder_post(
                    f"hashtag_search_{i}",
                    "unknown",
                )
                post_data["hashtags"] = [hashtag]

                json_content = json.dumps(post_data, indent=2).encode("utf-8")

                yield ScrapedItem(
                    content=json_content,
                    identifier=f"hashtag_{hashtag}_{i}.json",
                    metadata=post_data,
                    content_type="application/json",
                )
                items_yielded += 1

        # User handle - try Bluesky first (no auth needed)
        elif input_value.startswith("@"):
            username = input_value[1:]

            # Try Bluesky public API
            try:
                import requests

                # Bluesky handle format: user.bsky.social or custom domain
                if "." not in username:
                    bsky_handle = f"{username}.bsky.social"
                else:
                    bsky_handle = username

                self.logger.info(
                    f"Fetching posts from Bluesky @{bsky_handle}",
                    module=self.MODULE_NAME,
                    action="FETCH_USER",
                )

                # Resolve handle to DID
                resolve_url = f"https://bsky.social/xrpc/com.atproto.identity.resolveHandle?handle={bsky_handle}"
                resolve_resp = requests.get(resolve_url, timeout=10)

                if resolve_resp.status_code == 200:
                    did = resolve_resp.json().get("did")

                    # Fetch user's posts
                    feed_url = f"https://bsky.social/xrpc/app.bsky.feed.getAuthorFeed?actor={did}&limit={limit}"
                    feed_resp = requests.get(feed_url, timeout=15)

                    if feed_resp.status_code == 200:
                        feed_data = feed_resp.json()
                        posts = feed_data.get("feed", [])

                        self.logger.info(
                            f"Found {len(posts)} posts from @{bsky_handle}",
                            module=self.MODULE_NAME,
                            action="FETCH_COMPLETE",
                        )

                        for post_item in posts[:limit]:
                            if items_yielded >= limit:
                                break

                            post = post_item.get("post", {})
                            record = post.get("record", {})

                            post_data = {
                                "id": post.get("uri", "").split("/")[-1],
                                "platform": "bluesky",
                                "source_url": f"https://bsky.app/profile/{bsky_handle}/post/{post.get('uri', '').split('/')[-1]}",
                                "content": record.get("text", ""),
                                "author": {
                                    "handle": bsky_handle,
                                    "did": did,
                                    "display_name": post.get("author", {}).get("displayName", ""),
                                },
                                "created_at": record.get("createdAt"),
                                "metrics": {
                                    "likes": post.get("likeCount", 0),
                                    "reposts": post.get("repostCount", 0),
                                    "replies": post.get("replyCount", 0),
                                },
                                "scraped_at": datetime.now().isoformat(),
                            }

                            json_content = json.dumps(post_data, indent=2).encode("utf-8")

                            yield ScrapedItem(
                                content=json_content,
                                identifier=f"bluesky_{bsky_handle}_{post_data['id']}.json",
                                metadata=post_data,
                                content_type="application/json",
                                source_url=post_data["source_url"],
                            )
                            items_yielded += 1
                    else:
                        self.logger.warning(
                            f"Could not fetch feed for @{bsky_handle}",
                            module=self.MODULE_NAME,
                        )
                else:
                    self.logger.warning(
                        f"Could not resolve Bluesky handle @{bsky_handle} - user may not exist on Bluesky",
                        module=self.MODULE_NAME,
                    )

            except Exception as e:
                self.logger.error(
                    f"Bluesky fetch failed: {e}",
                    module=self.MODULE_NAME,
                )

        # Reddit - r/subreddit
        elif input_value.startswith("r/"):
            subreddit = input_value[2:]
            for item in self._fetch_reddit(subreddit, limit):
                yield item
                items_yielded += 1

        # File with list of URLs/handles
        elif input_value.startswith("file:"):
            file_path = Path(input_value[5:])

            with open(file_path, "r") as f:
                for line in f:
                    if items_yielded >= limit:
                        break
                    line = line.strip()
                    if line:
                        for item in self.fetch(line, 1):
                            yield item
                            items_yielded += 1
                            if items_yielded >= limit:
                                break

        # Search term
        else:
            self.logger.info(
                f"Would search for '{input_value}' (requires API credentials)",
                module=self.MODULE_NAME,
            )

    def get_hash(self, item: ScrapedItem) -> str:
        """Generate hash from post ID for duplicate detection."""
        post_id = item.metadata.get("id", "")
        platform = item.metadata.get("platform", "")
        canonical = f"{platform}:{post_id}"
        return HashManager.sha256_content(canonical.encode("utf-8"))

    def get_filename(self, item: ScrapedItem) -> str:
        """Generate filename from platform and post ID."""
        return self.validator.sanitize_filename(item.identifier)
