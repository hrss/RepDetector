import os
import sys
import requests


def is_paying_user(firebase_uid: str) -> bool:
    if not firebase_uid:
        return False

    api_key = "sk_GVboybEysXNLUGpOcJkfvmieGUPEE"
    if not api_key:
        print("REVENUE_CAT_KEY is not set")
        return False

    url = f"https://api.revenuecat.com/v1/subscribers/{firebase_uid}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    response = requests.get(url, headers=headers, timeout=5)

    if response.status_code == 404:
        return False

    response.raise_for_status()

    data = response.json()

    entitlements = (
        data.get("subscriber", {})
        .get("entitlements", {})
    )

    for entitlement in entitlements.values():
        expires_date = entitlement.get("expires_date")

        # RevenueCat uses null for lifetime/non-expiring entitlements
        if expires_date is None:
            return True

        # Simple string comparison works for ISO-8601 UTC timestamps
        from datetime import datetime, timezone

        expires_at = datetime.fromisoformat(
            expires_date.replace("Z", "+00:00")
        )

        if expires_at > datetime.now(timezone.utc):
            return True

    return False


if __name__ == "__main__":


    firebase_uid = "D8vpURAWkSMB0OvPLTHOjYQz0lJ3"

    try:
        paying = is_paying_user(firebase_uid)
        print(f"Paying user: {paying}")
    except requests.HTTPError as e:
        print(f"API returned an error: {e}")
        if e.response is not None:
            print(e.response.text)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)